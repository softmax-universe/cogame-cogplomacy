## Claude-backed decision making for Cogplomacy. Each seat's policy is just
## a prompt: the game server composes the seat's view (the whole board, the
## ownership table, two years of history, the press it received, its notes)
## plus that seat's prompt and asks Claude what it writes and what it
## orders.
##
## All seven powers decide SIMULTANEOUSLY by rule, so every phase fires its
## requests as ONE parallel batch (`curly.makeRequests`) — never seat by
## seat. An invalid reply joins one retry batch carrying the rejection reason;
## if a model seat still fails, the episode fails explicitly rather than
## substituting a scripted fallback.
##
## Credentials, in order of preference:
##   COWORLD_LLM_ENDPOINT            - hosted sidecar
##   Bedrock bearer token            - local play
##   ANTHROPIC_API_KEY               - the key itself
##   ANTHROPIC_API_KEY_URI           - a URI holding the key
## Explicit scripted seats (`PLAYER_SCRIPTED=expander` or `hedgehog`) run
## deterministically without an LLM; non-scripted seats require a configured
## LLM endpoint/credential.

import
  std/[algorithm, json, options, os, strutils, times, unicode],
  bitworld/runtime,
  curly,
  prompts,
  sim,
  training

export sim

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"
  BedrockAnthropicVersion = "bedrock-2023-05-31"
  Far = high(int) div 4

type
  DecisionFailure* = object of CogplomacyError
    seat*: int

  ScriptKind* = enum
    skNone = "none"
    skExpander = "expander"
    skHedgehog = "hedgehog"

  Decision* = object
    scripted*: bool
    broadcast*: string
    letters*: seq[Letter]
    pledges*: seq[Pledge]
    orders*: seq[string]
    retreats*: seq[string]
    adjustments*: seq[string]
    notes*: string

  LlmTransport = enum
    ltNone, ltSidecar, ltBedrock, ltAnthropic

  LlmClient* = ref object
    curl: Curly
    transport: LlmTransport
    apiKey: string
    sidecarEndpoint: string
    bedrockEndpoint: string
    bedrockModels: seq[string]
    bedrockModel: int
    bedrockToken: string
    model: string
    maxOutputTokens: int
    timeoutSeconds: int
    disabled*: bool
    nativeAttempts*: seq[seq[DecisionAttempt]]

proc parseScriptKind*(text: string): ScriptKind =
  ## Canonical PLAYER_SCRIPTED values: "" (LLM-driven), "expander", "hedgehog".
  let cleaned = text.strip().toLowerAscii()
  case cleaned
  of "": skNone
  of "expander": skExpander
  of "hedgehog": skHedgehog
  else:
    raise newException(CogplomacyError,
      "unknown PLAYER_SCRIPTED value: " & cleanText(oneLine(text), 64))

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    result = readCogameUri(uri, "ANTHROPIC_API_KEY_URI").strip()
  except CatchableError as error:
    echo "cogplomacy llm: failed to fetch ANTHROPIC_API_KEY_URI: ", error.msg
    result = ""

proc bedrockModelIds(): seq[string] =
  ## Bedrock inference-profile candidates, tried in order. BEDROCK_MODEL
  ## pins a single id; without it, fall through this list — model access is
  ## a per-account Marketplace subscription, so an id that works in one
  ## account 403s in another. Haiku leads: hosted Bedrock capacity is shared
  ## account-wide and the sonnet profiles run out of daily tokens first.
  let pinned = getEnv("BEDROCK_MODEL").strip()
  if pinned.len > 0:
    return @[pinned]
  @[
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  ]

proc tryNextBedrockModel(client: LlmClient, why: string): bool =
  if client.transport != ltBedrock or
      client.bedrockModel + 1 >= client.bedrockModels.len:
    return false
  client.bedrockModel.inc
  echo "cogplomacy llm: ", client.bedrockModels[client.bedrockModel - 1],
    " unusable (", why, "); falling back to ",
    client.bedrockModels[client.bedrockModel]
  true

proc bedrockUrl(client: LlmClient): string =
  client.bedrockEndpoint & "/model/" &
    client.bedrockModels[client.bedrockModel] & "/invoke"

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    result.model = getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    result.curl = newCurly()
    return
  let bedrockEndpoint = getEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").strip()
  let bedrockToken = getEnv("AWS_BEARER_TOKEN_BEDROCK").strip()
  if bedrockEndpoint.len > 0 or bedrockToken.len > 0:
    let region = getEnv("AWS_REGION",
      getEnv("AWS_DEFAULT_REGION", "us-west-2"))
    let endpoint =
      if bedrockEndpoint.len > 0: bedrockEndpoint
      else: "https://bedrock-runtime." & region & ".amazonaws.com"
    result.transport = ltBedrock
    result.bedrockEndpoint = endpoint.strip(chars = {'/'}, leading = false)
    result.bedrockModels = bedrockModelIds()
    result.bedrockToken = bedrockToken
    result.curl = newCurly()
    echo "cogplomacy llm: bedrock transport, model ",
      result.bedrockModels[result.bedrockModel],
      ", url ", result.bedrockUrl
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "cogplomacy llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    result.disabled = true
    echo "cogplomacy llm: no LLM credentials configured (scripted seats only)"

# ---- Scripted baselines -----------------------------------------------------

proc targetCentres(sim: Sim, power: int): seq[int] =
  for province in SupplyCentres:
    if sim.board.ownerOf(province) != power:
      result.add(province)

proc armyGoalDistance(sim: Sim, power: int): seq[int] =
  ## Multi-source BFS on the army graph: hops from every province to the
  ## nearest supply centre this power does NOT own.
  result = newSeq[int](NumProvinces)
  for index in 0 ..< NumProvinces:
    result[index] = Far
  var queue: seq[int]
  for province in sim.targetCentres(power):
    if Provinces[province].kind != pkSea:
      result[province] = 0
      queue.add(province)
  var head = 0
  while head < queue.len:
    let node = queue[head]
    inc head
    for next in ArmyAdj[node]:
      if result[next] == Far:
        result[next] = result[node] + 1
        queue.add(next)

proc fleetGoalDistance(sim: Sim, power: int): seq[int] =
  ## The same on the fleet-node graph, indexed by fleet node.
  result = newSeq[int](NumFleetNodes)
  for index in 0 ..< NumFleetNodes:
    result[index] = Far
  var queue: seq[int]
  for province in sim.targetCentres(power):
    for node in fleetNodesOf(province):
      if result[node] == Far:
        result[node] = 0
        queue.add(node)
  var head = 0
  while head < queue.len:
    let node = queue[head]
    inc head
    for next in FleetAdj[node]:
      if result[next] == Far:
        result[next] = result[node] + 1
        queue.add(next)

type ExpanderTuning* = object
  ## The two numbers in the expander baseline, kept as data so
  ## `tools/tune_baselines.nim` can sweep them rather than anyone guessing
  ## them. The shipped values are the design note's.
  springHomePenalty*: int   ## ranks a Spring move pays to leave a home centre
  supportFromRank*: int     ## lowest rank at which a unit supports instead

const DefaultExpanderTuning* = ExpanderTuning(springHomePenalty: 1,
  supportFromRank: 2)

type MoveOption = object
  dest: int
  coast: string
  rank: int    ## with the Spring home-centre penalty, for ordering
  base: int    ## the note's (a)/(b)/(c)/(d) before that penalty

proc rankMove(sim: Sim, power: int, unit: Unit, dest: int,
    originDist, destDist: int): int =
  ## (a) unowned neutral centre, (b) a rival's unoccupied centre,
  ## (c) closes the distance, (d) anything else.
  let owner = sim.board.ownerOf(dest)
  let occupied = sim.board.unitAt(dest) >= 0
  if Provinces[dest].isCentre and owner < 0:
    return 0
  if Provinces[dest].isCentre and owner >= 0 and owner != power and
      not occupied:
    return 1
  if destDist < originDist:
    return 2
  3

proc expanderOrders*(sim: Sim, power: int,
    tuning = DefaultExpanderTuning): seq[string] =
  let armyDist = sim.armyGoalDistance(power)
  let fleetDist = sim.fleetGoalDistance(power)
  var mine = sim.unitsOf(power)
  mine.sort(proc (a, b: Unit): int =
    cmp(provinceCode(a.province), provinceCode(b.province)))
  var claimed: seq[int]
  var chosen: seq[tuple[unit: Unit, dest: int, coast: string, rank: int]]
  for unit in mine:
    let originDist =
      if unit.kind == ukArmy: armyDist[unit.province]
      else: fleetDist[unitNode(unit)]
    var options: seq[MoveOption]
    if unit.kind == ukArmy:
      for dest in ArmyAdj[unit.province]:
        let base = sim.rankMove(power, unit, dest, originDist, armyDist[dest])
        var rank = base
        ## In Spring, a move that vacates an owned, unoccupied home centre
        ## drops one rank; in Fall the centres are taken regardless.
        if sim.season == seSpring and Provinces[unit.province].isCentre and
            Provinces[unit.province].homePower == power and
            sim.board.ownerOf(unit.province) == power:
          rank += tuning.springHomePenalty
        options.add(MoveOption(dest: dest, coast: "", rank: rank, base: base))
    else:
      for node in FleetAdj[unitNode(unit)]:
        let dest = FleetNodeProvince[node]
        let base = sim.rankMove(power, unit, dest, originDist, fleetDist[node])
        var rank = base
        if sim.season == seSpring and Provinces[unit.province].isCentre and
            Provinces[unit.province].homePower == power and
            sim.board.ownerOf(unit.province) == power:
          rank += tuning.springHomePenalty
        options.add(MoveOption(dest: dest, coast: FleetNodeCoast[node],
          rank: rank, base: base))
    options.sort(proc (a, b: MoveOption): int =
      if a.rank != b.rank: cmp(a.rank, b.rank)
      elif a.dest != b.dest: cmp(provinceCode(a.dest), provinceCode(b.dest))
      else: cmp(a.coast, b.coast))
    var picked = MoveOption(dest: -1, coast: "", rank: 3, base: 3)
    for option in options:
      if option.rank > 2:
        break
      if option.dest in claimed:
        continue
      ## Never walk into one of our own units that is staying put.
      let sitting = sim.board.unitAt(option.dest)
      if sitting >= 0 and sim.board.units[sitting].power == power:
        continue
      picked = option
      break
    if picked.dest >= 0:
      claimed.add(picked.dest)
    chosen.add((unit, picked.dest, picked.coast, picked.base))

  ## A unit whose own best option is only rank (c) — a move that merely
  ## closes the distance — or (d) supports one of ours that is moving
  ## instead, whenever it can reach that destination. Only units that are
  ## really moving are supported, so the baseline never writes a void
  ## support.
  var moving: seq[bool]
  for entry in chosen:
    moving.add(entry.dest >= 0 and entry.rank < tuning.supportFromRank)
  for index, entry in chosen:
    if not moving[index]:
      var helped = false
      for other in 0 ..< chosen.len:
        if other == index or not moving[other]:
          continue
        if not canReach(entry.unit, chosen[other].dest):
          continue
        result.add(unitLabel(entry.unit) & " S " &
          $chosen[other].unit.kind & " " &
          provinceCode(chosen[other].unit.province) & " - " &
          provinceCode(chosen[other].dest))
        helped = true
        break
      if helped:
        continue
      if entry.dest >= 0:
        moving[index] = true
    if moving[index]:
      result.add(unitLabel(entry.unit) & " - " & provinceCode(entry.dest) &
        (if entry.coast.len > 0: "/" & entry.coast else: ""))
    else:
      result.add(unitLabel(entry.unit) & " H")

proc hedgehogOrders(sim: Sim, power: int): seq[string] =
  ## The wall: everything holds, and a unit next to one of our own units
  ## sitting on a centre we own props it up instead.
  var mine = sim.unitsOf(power)
  mine.sort(proc (a, b: Unit): int =
    cmp(provinceCode(a.province), provinceCode(b.province)))
  for unit in mine:
    var neighbours: seq[Unit]
    for other in mine:
      if sameUnit(other, unit):
        continue
      if not Provinces[other.province].isCentre:
        continue
      if sim.board.ownerOf(other.province) != power:
        continue
      if canReach(unit, other.province):
        neighbours.add(other)
    neighbours.sort(proc (a, b: Unit): int =
      cmp(provinceCode(a.province), provinceCode(b.province)))
    if neighbours.len > 0:
      result.add(unitLabel(unit) & " S " & $neighbours[0].kind & " " &
        provinceCode(neighbours[0].province))
    else:
      result.add(unitLabel(unit) & " H")

proc scriptedRetreats(sim: Sim, power: int, kind: ScriptKind): seq[string] =
  let armyDist = sim.armyGoalDistance(power)
  let fleetDist = sim.fleetGoalDistance(power)
  for item in sim.dislodged:
    if item.unit.power != power:
      continue
    let options = retreatDestinations(sim.board, item.unit,
      item.attackerFrom, sim.standoffs, sim.dislodged)
    var best = -1
    var bestCoast = ""
    var bestScore = Far
    for option in options:
      var score = Far
      if kind == skHedgehog:
        for home in HomeCentres[power]:
          if sim.board.ownerOf(home) != power:
            continue
          let distances =
            if item.unit.kind == ukArmy: bfsDistance(option.province, false)
            else: bfsDistance(fleetNode(option.province, option.coast), true)
          if distances[home] < score:
            score = distances[home]
      else:
        score =
          if item.unit.kind == ukArmy: armyDist[option.province]
          else: fleetDist[fleetNode(option.province, option.coast)]
      if score < bestScore or (score == bestScore and best >= 0 and
          provinceCode(option.province) < provinceCode(best)):
        bestScore = score
        best = option.province
        bestCoast = option.coast
    if best < 0:
      result.add(unitLabel(item.unit) & " - D")
    else:
      result.add(unitLabel(item.unit) & " - " & provinceCode(best) &
        (if bestCoast.len > 0: "/" & bestCoast else: ""))

proc scriptedBuilds(sim: Sim, power: int, kind: ScriptKind): seq[string] =
  let delta = sim.centresOfPower(power) - sim.unitCount(power)
  if delta == 0:
    return
  if delta < 0:
    let owed = -delta
    var removed: seq[int]
    while removed.len < owed:
      let index = sim.civilDisorderPick(power, removed)
      if index < 0:
        break
      removed.add(sim.board.units[index].province)
      result.add("DISBAND " & unitLabel(sim.board.units[index]))
    return
  var fleets = 0
  var armies = 0
  for unit in sim.board.units:
    if unit.power != power:
      continue
    if unit.kind == ukFleet: inc fleets else: inc armies
  let sites = buildSites(sim.board, power)
  var provinces: seq[int]
  for site in sites:
    if site.province notin provinces:
      provinces.add(site.province)
  var built = 0
  for province in provinces:
    if built >= delta:
      break
    ## One build per centre, and the kind is decided once for the centre: a
    ## fleet when it is coastal and the power holds fewer fleets than armies,
    ## otherwise an army. Of a split coast's sites take the one with the most
    ## water to move on, which is what makes `STP` build `F STP/SC`.
    var chosen = -1
    if kind != skHedgehog and fleets < armies:
      for index, site in sites:
        if site.province != province or site.kind != ukFleet:
          continue
        if chosen < 0 or FleetAdj[fleetNode(province, site.coast)].len >
            FleetAdj[fleetNode(province, sites[chosen].coast)].len:
          chosen = index
    if chosen < 0:
      for index, site in sites:
        if site.province == province and site.kind == ukArmy and chosen < 0:
          chosen = index
    if chosen < 0:
      continue
    let site = sites[chosen]
    result.add("BUILD " & $site.kind & " " & provinceCode(site.province) &
      (if site.coast.len > 0: "/" & site.coast else: ""))
    if site.kind == ukFleet: inc fleets else: inc armies
    inc built

proc scriptedDecision*(sim: Sim, seat: int, kind: ScriptKind): Decision =
  ## Rule-based baseline for `seat`. Always legal; never talks or notes.
  let power = sim.powerOf[seat]
  result.scripted = true
  case sim.phase
  of pkPress:
    discard
  of pkOrders:
    result.orders =
      if kind == skHedgehog: sim.hedgehogOrders(power)
      else: sim.expanderOrders(power)
  of pkRetreats:
    result.retreats = sim.scriptedRetreats(power, kind)
  of pkBuilds:
    result.adjustments = sim.scriptedBuilds(power, kind)
  of pkDone:
    discard

# ---- Prompt building --------------------------------------------------------

proc seasonWord(season: Season): string =
  case season
  of seSpring: "SPRING"
  of seFall: "FALL"
  of seWinter: "WINTER"

proc systemPrompt*(sim: Sim, seat: int, selection: PromptSelection): string =
  ## The in-game name space is anonymous: a seat is only ever a POWER. No
  ## policy name, player name or slot number is ever interpolated here.
  let power = sim.powerOf[seat]
  var others: seq[string]
  for other in 0 ..< Powers:
    if other != power:
      others.add(PowerNames[other])
  let othersText = others[0 ..< others.high].join(", ") & " and " & others[^1]
  selection.introduction(PowerNames[power], othersText) & """
""" & selection.rulesBriefing() & """OUTPUT FORMAT: reply with ONLY one JSON object (a standard outer ```json ... ```
fence is also accepted), nothing else — no analysis, no explanation, and no text
before or after the object."""

proc operatorBlock(prompt: string): string =
  if prompt.len == 0:
    return ""
  "GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never above the " &
    "rules; always reply in the requested format):\n" & prompt & "\n\n"

proc boardBlock(sim: Sim): string =
  var lines: seq[string]
  for power in 0 ..< Powers:
    var pieces: seq[string]
    for unit in sim.board.units:
      if unit.power == power:
        pieces.add($unit.kind & " " & provinceName(unit.province) &
          (if unit.coast.len > 0: " (" & unit.coast & ")" else: ""))
    pieces.sort(system.cmp[string])
    lines.add(PowerNames[power] & " — " & $sim.centresOfPower(power) &
      " centres, " & $sim.unitCount(power) & " units: " &
      (if pieces.len > 0: pieces.join("; ") else: "(no units)"))
  "THE BOARD:\n" & lines.join("\n") & "\n\n"

proc ownershipBlock(sim: Sim): string =
  var lines: seq[string]
  for slot in 0 ..< NumCentres:
    let province = SupplyCentres[slot]
    let owner = sim.board.owner[slot]
    lines.add(provinceName(province) & " (" & provinceCode(province) & "): " &
      (if owner < 0: "neutral" else: PowerNames[owner]))
  "SUPPLY CENTRES (34):\n" & lines.join("; ") & "\n\n"

proc historyBlock(sim: Sim): string =
  if sim.history.len == 0:
    return "RECENT ORDERS: (none yet)\n\n"
  var blocks: seq[string]
  for record in sim.history:
    blocks.add(seasonWord(record.season) & " " & $record.year & "\n" &
      record.lines.join("\n"))
  "RECENT ORDERS AND RESULTS:\n" & blocks.join("\n\n") & "\n\n"

proc pressBlock(sim: Sim, power: int): string =
  var lines: seq[string]
  for letter in sim.pressLast:
    if letter.toPower < 0:
      lines.add("[last phase] " & PowerNames[letter.fromPower] &
        " broadcast: \"" & letter.text & "\"")
    elif letter.toPower == power:
      lines.add("[last phase] " & PowerNames[letter.fromPower] &
        " wrote to you: \"" & letter.text & "\"")
  for letter in sim.press:
    if letter.toPower < 0:
      lines.add(PowerNames[letter.fromPower] & " broadcast: \"" &
        letter.text & "\"")
    elif letter.toPower == power:
      lines.add(PowerNames[letter.fromPower] & " wrote to you: \"" &
        letter.text & "\"")
  for pledge in sim.pledges:
    if pledge.fromPower == power:
      continue
    if pledge.toPower < 0 or pledge.toPower == power:
      lines.add(PowerNames[pledge.fromPower] & " pledged " & $pledge.kind &
        (if pledge.province >= 0: " (" & provinceName(pledge.province) & ")"
         else: "") &
        (if pledge.toPower < 0: " to everyone" else: " to you"))
  "PRESS YOU HAVE RECEIVED:\n" &
    (if lines.len > 0: lines.join("\n") else: "(none)") & "\n\n"

proc legalOrdersBlock(sim: Sim, power: int): string =
  var blocks: seq[string]
  var mine = sim.unitsOf(power)
  mine.sort(proc (a, b: Unit): int =
    cmp(provinceCode(a.province), provinceCode(b.province)))
  for unit in mine:
    blocks.add(unitLabel(unit) & " in " & provinceName(unit.province) & ":\n  " &
      legalOrders(sim.board, unit).join("\n  "))
  "YOUR UNITS AND EVERY LEGAL ORDER:\n" & blocks.join("\n") & "\n\n"

proc retreatBlock(sim: Sim, power: int): string =
  var lines: seq[string]
  for item in sim.dislodged:
    if item.unit.power != power:
      continue
    lines.add(unitLabel(item.unit) & " was dislodged from " &
      provinceName(item.unit.province) & " by an attack out of " &
      provinceName(item.attackerFrom) & ". Legal: " &
      legalRetreats(sim.board, item.unit, item.attackerFrom,
        sim.standoffs, sim.dislodged).join(", ") & " (disband).")
  lines.join("\n") & "\n\n"

proc buildBlock(sim: Sim, power: int): string =
  let delta = sim.centresOfPower(power) - sim.unitCount(power)
  if delta > 0:
    var sites: seq[string]
    for option in legalBuilds(sim.board, power):
      sites.add(option)
    return "You own " & $sim.centresOfPower(power) & " centres and have " &
      $sim.unitCount(power) & " units: build " & $delta & ". Legal builds: " &
      (if sites.len > 0: sites.join(", ") else: "(none — the entitlement is waived)") &
      "\n\n"
  "You own " & $sim.centresOfPower(power) & " centres and have " &
    $sim.unitCount(power) & " units: disband " & $(-delta) &
    ". Legal disbands: " & legalDisbands(sim.board, power).join(", ") & "\n\n"

proc userPrompt*(sim: Sim, seat: int, prompt: string): string =
  let power = sim.powerOf[seat]
  let head = seasonWord(sim.season) & " " & $sim.year
  result.add(head & " — " & ($sim.phase).toUpperAscii() & ". Year " &
    $(sim.yearsPlayed + 1) & " of " & $sim.config.years & ".\n\n")
  result.add(sim.boardBlock())
  result.add(sim.ownershipBlock())
  result.add(sim.historyBlock())
  if sim.config.press:
    result.add(sim.pressBlock(power))
  result.add("YOUR NOTES:\n" &
    (if sim.notes[seat].len > 0: sim.notes[seat] else: "(none)") & "\n\n")
  result.add(operatorBlock(prompt))
  case sim.phase
  of pkPress:
    result.add("Reply with ONLY {\"broadcast\":\"…\"," &
      "\"letters\":[{\"to\":\"ITALY\",\"text\":\"…\"}]," &
      "\"pledges\":[{\"to\":\"ITALY\",\"kind\":\"peace\"}],\"notes\":\"…\"}" &
      " — broadcast at most " & $MaxBroadcastLen & " characters, at most " &
      $MaxLetters & " letters of at most " & $MaxLetterLen &
      " characters each (one per power), at most " & $MaxPledges &
      " pledges, notes at most " & $MaxNotesLen & " characters. A pledge " &
      "kind is peace, keepout (with a \"province\" code) or support. A " &
      "pledge is the only promise spectators can see you break; free text " &
      "is never checked.")
  of pkOrders:
    result.add(sim.legalOrdersBlock(power))
    result.add("Reply with ONLY {\"orders\":[\"A PAR - BUR\"," &
      "\"F BRE S A PAR - BUR\"],\"notes\":\"…\"} — exactly one legal order per " &
      "unit, copied from the list above.")
  of pkRetreats:
    result.add(sim.retreatBlock(power))
    result.add("Reply with ONLY {\"retreats\":[\"A VIE - TYR\"]," &
      "\"notes\":\"…\"} — exactly one legal entry per dislodged unit; " &
      "\"A VIE - D\" disbands it.")
  of pkBuilds:
    result.add(sim.buildBlock(power))
    result.add("Reply with ONLY {\"adjustments\":[\"BUILD A PAR\"]," &
      "\"notes\":\"…\"} (or \"DISBAND F BRE\" when you must remove units; " &
      "use [] only when waiving builds or when no build sites are available).")
  of pkDone:
    discard

# ---- Reply parsing ----------------------------------------------------------

proc extractJsonObject*(text: string): JsonNode =
  ## Parses a JSON object, accepting either raw JSON or a single standard
  ## outer Markdown JSON fence (` ```json ... ``` ` or ` ``` ... ``` `).
  ## Surrounding prose is rejected.
  var body = text.strip()
  if body.startsWith("```"):
    let firstNl = body.find('\n')
    if firstNl < 0:
      raise newException(CogplomacyError, "malformed markdown fence in response: " &
        cleanText(oneLine(text), 160))
    let header = body[3 ..< firstNl].strip().toLowerAscii()
    if header.len > 0 and header != "json":
      raise newException(CogplomacyError, "unsupported markdown fence language: " &
        cleanText(oneLine(header), 32))
    let afterHeader = body[firstNl + 1 .. ^1]
    if not afterHeader.endsWith("```"):
      raise newException(CogplomacyError, "unclosed markdown fence in response: " &
        cleanText(oneLine(text), 160))
    let inner = afterHeader[0 ..< afterHeader.len - 3]
    if inner.len > 0 and inner[^1] notin {'\n', '\r'}:
      raise newException(CogplomacyError, "malformed closing markdown fence in response: " &
        cleanText(oneLine(text), 160))
    body = inner.strip()
  if body.len < 2 or body[0] != '{' or body[^1] != '}':
    raise newException(CogplomacyError, "expected JSON object in response: " &
      cleanText(oneLine(text), 160))
  try:
    result = parseJson(body)
  except CatchableError as error:
    raise newException(CogplomacyError, "invalid JSON in response: " & error.msg)
  if result.kind != JObject:
    raise newException(CogplomacyError, "reply is not a JSON object")

proc parseNotesField(payload: JsonNode): string =
  let node = payload{"notes"}
  if node.isNil:
    return ""
  if node.kind != JString:
    raise newException(CogplomacyError, "notes must be a string")
  cleanText(node.getStr(), MaxNotesLen)

proc strictStringList(node: JsonNode, fieldName: string, limit: int): seq[string] =
  if node.isNil or node.kind != JArray:
    raise newException(CogplomacyError, "no " & fieldName & " array in reply")
  if node.len > limit:
    raise newException(CogplomacyError, fieldName & " exceeds limit of " & $limit)
  for item in node:
    if item.kind != JString:
      raise newException(CogplomacyError, fieldName & " entries must be strings")
    let raw = oneLine(item.getStr()).strip()
    if raw.runeLen > MaxOrderLen:
      raise newException(CogplomacyError, fieldName & " entry exceeds max length: " &
        cleanText(raw, MaxOrderLen))
    result.add(raw)

proc parsePress(sim: Sim, power: int, payload: JsonNode): Decision =
  let broadcast = payload{"broadcast"}
  let letters = payload{"letters"}
  let pledges = payload{"pledges"}
  if not broadcast.isNil and broadcast.kind != JString:
    raise newException(CogplomacyError, "broadcast must be a string")
  if not letters.isNil and letters.kind != JArray:
    raise newException(CogplomacyError, "letters must be an array")
  if not pledges.isNil and pledges.kind != JArray:
    raise newException(CogplomacyError, "pledges must be an array")
  if broadcast.isNil and letters.isNil:
    raise newException(CogplomacyError, "no broadcast or letters in reply")
  if not broadcast.isNil:
    result.broadcast = cleanText(oneLine(broadcast.getStr()), MaxBroadcastLen)
  if not letters.isNil:
    if letters.len > MaxLetters:
      raise newException(CogplomacyError, "too many letters (max " & $MaxLetters & ")")
    var seenRecipients: seq[int]
    for item in letters:
      if item.kind != JObject:
        raise newException(CogplomacyError, "letter entry must be an object")
      let toNode = item{"to"}
      let textNode = item{"text"}
      if toNode.isNil or toNode.kind != JString:
        raise newException(CogplomacyError, "letter.to must be a string")
      if textNode.isNil or textNode.kind != JString:
        raise newException(CogplomacyError, "letter.text must be a string")
      let toText = toNode.getStr().strip().toUpperAscii()
      var to = -1
      if toText != "ALL":
        to = powerByName(toText)
        if to < 0:
          raise newException(CogplomacyError, "unknown letter recipient: " & toText)
        if to == power:
          raise newException(CogplomacyError, "cannot send private letter to self")
        if to in seenRecipients:
          raise newException(CogplomacyError, "duplicate letter recipient: " & toText)
        seenRecipients.add(to)
      result.letters.add(Letter(fromPower: power, toPower: to,
        text: cleanText(oneLine(textNode.getStr()), MaxLetterLen)))
  if not pledges.isNil:
    if pledges.len > MaxPledges:
      raise newException(CogplomacyError, "too many pledges (max " & $MaxPledges & ")")
    for item in pledges:
      if item.kind != JObject:
        raise newException(CogplomacyError, "pledge entry must be an object")
      let kindNode = item{"kind"}
      if kindNode.isNil or kindNode.kind != JString:
        raise newException(CogplomacyError, "pledge.kind must be a string")
      var kind: PledgeKind
      case kindNode.getStr().strip().toLowerAscii()
      of "peace": kind = plPeace
      of "keepout": kind = plKeepOut
      of "support": kind = plSupport
      else:
        raise newException(CogplomacyError, "unknown pledge kind: " & kindNode.getStr())
      let toNode = item{"to"}
      if not toNode.isNil and toNode.kind != JString:
        raise newException(CogplomacyError, "pledge.to must be a string")
      let toText = if toNode.isNil: "" else: toNode.getStr().strip().toUpperAscii()
      var to = -1
      if toText != "ALL" and toText.len > 0:
        to = powerByName(toText)
        if to < 0:
          raise newException(CogplomacyError, "unknown pledge recipient: " & toText)
        if kind != plKeepOut and to == power:
          raise newException(CogplomacyError, "cannot pledge to self")
      var province = -1
      if kind == plKeepOut:
        let provNode = item{"province"}
        if provNode.isNil or provNode.kind != JString:
          raise newException(CogplomacyError, "keepout pledge requires province code")
        province = provinceByCode(provNode.getStr())
        if province < 0:
          raise newException(CogplomacyError, "unknown keepout province: " & provNode.getStr())
      result.pledges.add(Pledge(fromPower: power, toPower: to, kind: kind,
        province: province))
  result.notes = parseNotesField(payload)

proc parseOrdersReply(sim: Sim, power: int, payload: JsonNode): Decision =
  let raws = strictStringList(payload{"orders"}, "orders", MaxOrdersPerReply)
  var claimed: seq[int]
  for raw in raws:
    let parsed = parseOrder(sim.board, power, raw)
    if parsed.illegal:
      raise newException(CogplomacyError, "illegal order (" & parsed.why & "): " & raw)
    if parsed.unit.province in claimed:
      raise newException(CogplomacyError, "duplicate order for unit " & unitLabel(parsed.unit))
    claimed.add(parsed.unit.province)
    result.orders.add(formatOrder(parsed))
  for unit in sim.board.units:
    if unit.power == power and unit.province notin claimed:
      raise newException(CogplomacyError, "missing order for unit " & unitLabel(unit))
  result.notes = parseNotesField(payload)

proc parseRetreatsReply(sim: Sim, power: int, payload: JsonNode): Decision =
  let raws = strictStringList(payload{"retreats"}, "retreats", MaxRetreatsPerReply)
  var handled: seq[int]
  for raw in raws:
    let parsed = parseRetreat(sim.board, power, raw, sim.dislodged, sim.standoffs)
    if not parsed.ok:
      raise newException(CogplomacyError, "illegal retreat (" & parsed.why & "): " & raw)
    if parsed.unit.province in handled:
      raise newException(CogplomacyError, "duplicate retreat for unit " & unitLabel(parsed.unit))
    handled.add(parsed.unit.province)
    result.retreats.add(unitLabel(parsed.unit) & " - " &
      (if parsed.to < 0: "D" else: provinceCode(parsed.to) &
        (if parsed.coast.len > 0: "/" & parsed.coast else: "")))
  for item in sim.dislodged:
    if item.unit.power == power and item.unit.province notin handled:
      raise newException(CogplomacyError, "missing retreat for dislodged unit " &
        unitLabel(item.unit))
  result.notes = parseNotesField(payload)

proc parseBuildsReply(sim: Sim, power: int, payload: JsonNode): Decision =
  let raws = strictStringList(payload{"adjustments"}, "adjustments", MaxAdjustmentsPerReply)
  let delta = sim.buildDelta(power)
  var used: seq[int]
  for raw in raws:
    let parsed = parseAdjustment(sim.board, power, raw)
    if not parsed.ok:
      raise newException(CogplomacyError, "illegal adjustment (" & parsed.why & "): " & raw)
    if delta > 0 and not parsed.build:
      raise newException(CogplomacyError, "expected BUILD adjustment, got: " & raw)
    if delta < 0 and parsed.build:
      raise newException(CogplomacyError, "expected DISBAND adjustment, got: " & raw)
    if parsed.unit.province in used:
      raise newException(CogplomacyError, "duplicate adjustment for province " &
        provinceCode(parsed.unit.province))
    used.add(parsed.unit.province)
    result.adjustments.add((if parsed.build: "BUILD " else: "DISBAND ") &
      unitLabel(parsed.unit))
  if delta > 0 and result.adjustments.len > delta:
    raise newException(CogplomacyError, "too many builds: " & $result.adjustments.len &
      " > " & $delta)
  if delta < 0 and result.adjustments.len != -delta:
    raise newException(CogplomacyError, "must disband exactly " & $(-delta) &
      " units, got " & $result.adjustments.len)
  result.notes = parseNotesField(payload)

proc parseDecision*(sim: Sim, seat: int, payload: JsonNode): Decision =
  ## Validates and parses a model reply for the current phase. Any structural
  ## error, illegal action, or missing required unit order raises
  ## `CogplomacyError` instead of silently repairing or synthesizing fallbacks.
  if payload.isNil or payload.kind != JObject:
    raise newException(CogplomacyError, "reply is not a JSON object")
  case sim.phase
  of pkPress: parsePress(sim, sim.powerOf[seat], payload)
  of pkOrders: parseOrdersReply(sim, sim.powerOf[seat], payload)
  of pkRetreats: parseRetreatsReply(sim, sim.powerOf[seat], payload)
  of pkBuilds: parseBuildsReply(sim, sim.powerOf[seat], payload)
  of pkDone: Decision()

proc decisionReply*(decision: Decision, phase: PhaseKind): JsonNode =
  case phase
  of pkPress:
    var letters = newJArray()
    for letter in decision.letters:
      letters.add(%*{"to": (if letter.toPower < 0: "ALL" else: PowerNames[letter.toPower]),
        "text": letter.text})
    var pledges = newJArray()
    for pledge in decision.pledges:
      pledges.add(%*{"kind": $pledge.kind,
        "to": (if pledge.toPower < 0: "ALL" else: PowerNames[pledge.toPower]),
        "province": (if pledge.province < 0: "" else: provinceCode(pledge.province))})
    %*{"broadcast": decision.broadcast, "letters": letters, "pledges": pledges,
      "notes": decision.notes}
  of pkOrders: %*{"orders": decision.orders, "notes": decision.notes}
  of pkRetreats: %*{"retreats": decision.retreats, "notes": decision.notes}
  of pkBuilds: %*{"adjustments": decision.adjustments, "notes": decision.notes}
  of pkDone: raise newException(ValueError, "completed phase has no decision")

proc executedReply*(event: GameEvent): JsonNode =
  var decision = Decision(broadcast: event.broadcast,
    pledges: event.pledges, orders: event.orders, notes: event.text)
  for letter in event.letters:
    if letter.toPower >= 0 or letter.text != event.broadcast:
      decision.letters.add(letter)
  case event.kind
  of evPress, evOrders:
    discard
  of evRetreat:
    for move in event.moves:
      decision.retreats.add(unitLabel(move.unit) & " - " &
        (if move.to < 0: "D" else: provinceCode(move.to) &
          (if move.coast.len > 0: "/" & move.coast else: "")))
  of evBuild:
    for adjustment in event.adjustments:
      decision.adjustments.add(adjustment.action.toUpperAscii() & " " & unitLabel(adjustment.unit))
  else:
    raise newException(ValueError, "event is not an accepted submission")
  decision.decisionReply(event.phaseKind)

# ---- Anthropic / Bedrock transport ------------------------------------------

proc requestFor(client: LlmClient, system, user: string, slot: int):
    tuple[url: string, headers: HttpHeaders, body: string] =
  var body = %*{
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }
  var headers: HttpHeaders
  if client.transport == ltSidecar and slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  headers["content-type"] = "application/json"
  if client.transport == ltBedrock:
    body["anthropic_version"] = %BedrockAnthropicVersion
    if client.bedrockToken.len > 0:
      headers["authorization"] = "Bearer " & client.bedrockToken
    result.url = client.bedrockUrl()
  elif client.transport == ltSidecar:
    body["model"] = %client.model
    headers["anthropic-version"] = AnthropicVersion
    result.url = client.sidecarEndpoint & "/v1/messages"
  else:
    body["model"] = %client.model
    ## Only the Claude 5 / Opus tiers accept an effort setting; Haiku 4.5
    ## rejects the whole request with a 400 if it is present.
    if "haiku" notin client.model and "4-5" notin client.model:
      body["output_config"] = %*{"effort": "low"}
    headers["x-api-key"] = client.apiKey
    headers["anthropic-version"] = AnthropicVersion
    result.url = AnthropicUrl
  result.headers = headers
  result.body = $body

proc textOf(client: LlmClient, response: Response, error, url: string):
    string =
  ## The text of one batched reply, or a CogplomacyError describing why
  ## there is none. Auth failures disable the client; model-access and
  ## throttle failures rotate the Bedrock model for the next batch.
  if error.len > 0:
    raise newException(CogplomacyError, "llm transport: " & error)
  if response.code == 401 or response.code == 403:
    let detail = cleanText(oneLine(response.body), 400)
    if "Model access is denied" in response.body and
        client.tryNextBedrockModel("no model access"):
      raise newException(CogplomacyError,
        "bedrock model access denied: " & detail)
    client.disabled = true
    raise newException(CogplomacyError,
      "llm auth failed (" & $response.code & ") at " & url & ": " & detail)
  if response.code == 429:
    let detail = cleanText(oneLine(response.body), 300)
    discard client.tryNextBedrockModel("throttled")
    raise newException(CogplomacyError, "llm throttled (429): " & detail)
  if response.code < 200 or response.code >= 300:
    raise newException(CogplomacyError, "anthropic error " & $response.code &
      ": " & cleanText(oneLine(response.body), 300))
  let payload = parseJson(response.body)
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(CogplomacyError, "anthropic refusal")
  for contentBlock in payload["content"]:
    if contentBlock{"type"}.getStr() == "text":
      result.add(contentBlock{"text"}.getStr())
  if payload{"stop_reason"}.getStr() == "max_tokens" and '{' notin result:
    raise newException(CogplomacyError, "reply cut off at max_tokens before " &
      "any JSON: " & cleanText(oneLine(result), 160))

proc decideAll*(
  client: LlmClient,
  selection: PromptSelection,
  sim: Sim,
  seats: seq[int],
  prompts: seq[string],
  scripted: seq[ScriptKind]
): seq[Decision] =
  ## One decision per seat in `seats`, in order, for the CURRENT phase.
  ## Every live model seat's request goes out in ONE parallel batch.
  ## Explicit scripted seats run deterministically; non-scripted model seats
  ## require an active LLM endpoint and raise `DecisionFailure` if no endpoint
  ## is configured or if the model reply still fails after one retry.
  result = newSeq[Decision](seats.len)
  let capture = getEnv(CogameSaveTrajectoryUriEnv).len > 0
  if capture: client.nativeAttempts = newSeq[seq[DecisionAttempt]](Seats)
  var open: seq[int]
  var lastErrors = newSeq[string](seats.len)
  for index, seat in seats:
    let kind = scripted[seat]
    if kind != skNone:
      result[index] = scriptedDecision(sim, seat, kind)
    elif client.disabled:
      var err = newException(DecisionFailure,
        "no LLM endpoint or credentials configured for model seat " & $seat)
      err.seat = seat
      raise err
    else:
      open.add(index)
  for attempt in 0 .. 1:
    if open.len == 0 or client.disabled:
      break
    let started = epochTime()
    var batch: RequestBatch
    var pendingEvidence: seq[DecisionAttempt]
    for index in open:
      let seat = seats[index]
      var user = sim.userPrompt(seat, prompts[seat])
      if attempt > 0:
        user.add("\n\nYour previous reply was rejected (" & lastErrors[index] &
          "). Respond with ONLY the requested valid JSON object.")
      let request = client.requestFor(systemPrompt(sim, seat, selection), user, seat)
      if capture:
        pendingEvidence.add(nativeAttempt(sim.config.players[seat].name,
          sim.events.len, seat, attempt, request.body))
      batch.post(request.url, request.headers, request.body, $index)
    let responses = client.curl.makeRequests(batch, client.timeoutSeconds)
    let latency = int((epochTime() - started) * 1000.0)
    var stillOpen: seq[int]
    for position, index in open:
      let seat = seats[index]
      var evidence: DecisionAttempt
      if capture: evidence = pendingEvidence[position]
      try:
        if capture:
          evidence.receiveNativeResponse(responses[position].response,
            responses[position].error, latency)
        let text = client.textOf(responses[position].response,
          responses[position].error, batch[position].url)
        let payload = extractJsonObject(text)
        result[index] = sim.parseDecision(seat, payload)
        if capture:
          evidence.parsedAction = result[index].decisionReply(sim.phase)
          var candidate = copy(payload)
          let defaults = if sim.phase == pkPress:
            %*{"broadcast": "", "letters": [], "pledges": [], "notes": ""}
            else: %*{"notes": ""}
          for key, value in defaults:
            if not candidate.hasKey(key): candidate[key] = value
          if sim.phase == pkPress and candidate["pledges"].kind == JArray:
            for pledge in candidate["pledges"]:
              if pledge.kind == JObject:
                if not pledge.hasKey("province"): pledge["province"] = %""
                if not pledge.hasKey("to"): pledge["to"] = %"ALL"
          evidence.accepted = candidate == evidence.parsedAction
          if not evidence.accepted:
            evidence.rejectionReason = some("native parser normalized the proposal")
      except CatchableError as error:
        if capture: evidence.rejectionReason = some(error.msg)
        lastErrors[index] = error.msg
        echo "cogplomacy llm: seat ", seat, " attempt ", attempt,
          " failed: ", error.msg
        stillOpen.add(index)
      if capture: client.nativeAttempts[seat].add(evidence)
    open = stillOpen
  if open.len > 0:
    let failedIndex = open[0]
    let seat = seats[failedIndex]
    let reason =
      if lastErrors[failedIndex].len > 0: lastErrors[failedIndex]
      else: "model decision failed for seat " & $seat
    var err = newException(DecisionFailure,
      "seat " & $seat & " failed: " & reason)
    err.seat = seat
    raise err
