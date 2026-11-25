import cogplomacy/prompts
import std/[json, os, strutils, unicode, unittest]
import cogplomacy/llm
import ../tools/tune_baselines

## The scripted baselines: bounded, always legal, and explicitly selected via
## PLAYER_SCRIPTED. Reply parsing is exercised here too: invalid replies and
## non-canonical aliases are rejected rather than silently repaired or fallen
## back to scripted decisions.

proc fixtureConfig(years = 4, seed = 0, press = true): GameConfig =
  result = defaultGameConfig()
  result.years = years
  result.seed = seed
  result.press = press
  result.sampled = true
  for index in 0 ..< Seats:
    result.players.add(PlayerConfig(name: "P" & $(index + 1)))
    result.tokens.add("token-" & $index)

proc playEpisode(config: GameConfig, kinds: seq[ScriptKind]): Sim =
  result = initSim(config)
  let client = newLlmClient(config)
  let prompts = newSeq[string](Seats)
  var guard = 0
  while not result.done and guard < 2000:
    inc guard
    let seats = result.pendingSeats()
    doAssert seats.len > 0, "no pending seats and the episode is not over"
    let decisions = client.decideAll(PromptSelection(), result, seats, prompts, kinds)
    for index, seat in seats:
      let decision = decisions[index]
      case result.phase
      of pkPress:
        result.applyPress(seat, decision.broadcast, decision.letters,
          decision.pledges, decision.notes, true)
      of pkOrders:
        result.applyOrders(seat, decision.orders, decision.notes, true)
      of pkRetreats:
        result.applyRetreats(seat, decision.retreats, decision.notes, true)
      of pkBuilds:
        result.applyBuilds(seat, decision.adjustments, decision.notes, true)
      of pkDone:
        discard

proc allOf(kind: ScriptKind): seq[ScriptKind] =
  for index in 0 ..< Seats:
    result.add(kind)

proc auditLegality(sim: Sim) =
  ## Every submitted order parses and is legal, every unit is ordered
  ## exactly once, and no power ever stands itself off.
  var unitsAt: seq[int]
  for event in sim.events:
    case event.kind
    of evPhase:
      unitsAt = @[]
      for unit in event.units:
        unitsAt.add(unit.province)
    of evOrders:
      check event.illegal.len == 0
      var ordered: seq[string]
      var destinations: seq[string]
      for order in event.orders:
        let parts = strutils.splitWhitespace(order)
        check parts.len >= 2
        check parts[1] notin ordered
        ordered.add(parts[1])
        if parts.len >= 4 and parts[2] == "-":
          ## No power may stand itself off: one destination per power.
          check parts[3] notin destinations
          destinations.add(parts[3])
    of evRetreat:
      for move in event.moves:
        check move.to >= -1
    of evBuild:
      for action in event.adjustments:
        check action.action in ["build", "disband"]
        if action.action == "build":
          check Provinces[action.unit.province].isCentre
          check Provinces[action.unit.province].homePower == event.power
    else:
      discard

proc auditCaps(sim: Sim) =
  for event in sim.events:
    check event.letters.len <= MaxLetters
    check event.pledges.len <= MaxPledges
    check event.orders.len <= MaxOrdersPerReply
    check event.moves.len <= MaxRetreatsPerReply
    check event.adjustments.len <= MaxAdjustmentsPerReply

suite "expander":
  test "seven expanders play legal, complete episodes for seeds 1..8":
    for seed in 1 .. 8:
      let sim = playEpisode(fixtureConfig(years = 4, seed = seed),
        allOf(skExpander))
      check sim.done
      check sim.reason == "complete"
      auditLegality(sim)
      auditCaps(sim)

  test "every unit is ordered exactly once, every phase":
    let sim = playEpisode(fixtureConfig(years = 3, seed = 3),
      allOf(skExpander))
    var boardUnits: seq[Unit]
    for event in sim.events:
      if event.kind == evPhase and event.phaseKind == pkOrders:
        boardUnits = event.units
      if event.kind == evOrders:
        var mine = 0
        for unit in boardUnits:
          if unit.power == event.power:
            inc mine
        check event.orders.len == mine

  test "builds and disbands match the delta exactly":
    let sim = playEpisode(fixtureConfig(years = 5, seed = 2),
      allOf(skExpander))
    var counts: seq[int]
    var unitCounts: seq[int]
    for event in sim.events:
      if event.kind == evCentres:
        counts = event.counts
      if event.kind == evPhase and event.phaseKind == pkBuilds:
        unitCounts = newSeq[int](Powers)
        for unit in event.units:
          inc unitCounts[unit.power]
      if event.kind == evBuild and counts.len == Powers and
          unitCounts.len == Powers:
        let delta = counts[event.power] - unitCounts[event.power]
        var builds = 0
        var disbands = 0
        for action in event.adjustments:
          if action.action == "build": inc builds else: inc disbands
        if delta > 0:
          check builds + event.waived == delta
          check disbands == 0
        elif delta < 0:
          check disbands == -delta
          check builds == 0

  test "a unit that would only close the distance supports a neighbour":
    ## RUH's best move of its own is HOL, which merely closes the distance —
    ## rank (c) — while BUR is taking a centre it can reach, so RUH backs
    ## that move instead of making its own.
    var sim = initSim(fixtureConfig(years = 1, press = false))
    let power = 2                       # France
    let seat = sim.seatOf[power]
    sim.board.units = @[
      Unit(power: power, kind: ukArmy, province: provinceByCode("BUR")),
      Unit(power: power, kind: ukArmy, province: provinceByCode("RUH")),
      Unit(power: 3, kind: ukArmy, province: provinceByCode("HOL"))
    ]
    for slot in 0 ..< NumCentres:
      sim.board.owner[slot] = power
    sim.board.owner[CentreIndex[provinceByCode("BEL")]] = 3
    sim.board.owner[CentreIndex[provinceByCode("HOL")]] = 3
    let orders = scriptedDecision(sim, seat, skExpander).orders
    check "A BUR - BEL" in orders
    check "A RUH S A BUR - BEL" in orders

  test "a vacant coastal home centre builds a fleet when fleets trail armies":
    var sim = initSim(fixtureConfig(years = 2, seed = 4))
    let power = 5                       # Russia, whose STP has two coasts
    let seat = sim.seatOf[power]
    sim.board.units = @[
      Unit(power: power, kind: ukArmy, province: provinceByCode("MOS")),
      Unit(power: power, kind: ukArmy, province: provinceByCode("WAR"))
    ]
    for slot in 0 ..< NumCentres:
      sim.board.owner[slot] = -1
    for code in ["MOS", "WAR", "STP"]:
      sim.board.owner[CentreIndex[provinceByCode(code)]] = power
    sim.season = seWinter
    sim.phase = pkBuilds
    sim.pending = @[seat]
    check sim.buildDelta(power) == 1
    let decision = scriptedDecision(sim, seat, skExpander)
    check decision.adjustments == @["BUILD F STP/SC"]
    sim.applyBuilds(seat, decision.adjustments, "", true)
    var fleets = 0
    for unit in sim.board.units:
      if unit.kind == ukFleet:
        inc fleets
    check fleets == 1

  test "the hedgehog builds an army in the same place":
    var sim = initSim(fixtureConfig(years = 2, seed = 4))
    let power = 5
    let seat = sim.seatOf[power]
    sim.board.units = @[
      Unit(power: power, kind: ukArmy, province: provinceByCode("MOS")),
      Unit(power: power, kind: ukArmy, province: provinceByCode("WAR"))
    ]
    for slot in 0 ..< NumCentres:
      sim.board.owner[slot] = -1
    for code in ["MOS", "WAR", "STP"]:
      sim.board.owner[CentreIndex[provinceByCode(code)]] = power
    sim.season = seWinter
    sim.phase = pkBuilds
    sim.pending = @[seat]
    check scriptedDecision(sim, seat, skHedgehog).adjustments ==
      @["BUILD A STP"]

suite "hedgehog":
  test "seven hedgehogs play legal, complete episodes":
    for seed in [1, 5, 9]:
      let sim = playEpisode(fixtureConfig(years = 4, seed = seed),
        allOf(skHedgehog))
      check sim.done
      check sim.reason == "complete"
      auditLegality(sim)

  test "the wall never grows past its start":
    let sim = playEpisode(fixtureConfig(years = 4, seed = 5),
      allOf(skHedgehog))
    for seat in 0 ..< Seats:
      check sim.centres(seat) <= 4

suite "a mixed table":
  test "four expanders and three hedgehogs still complete":
    var kinds: seq[ScriptKind]
    for index in 0 ..< Seats:
      kinds.add(if index < 4: skExpander else: skHedgehog)
    let sim = playEpisode(fixtureConfig(years = 4, seed = 7), kinds)
    check sim.done
    check sim.reason == "complete"
    auditLegality(sim)
    ## The greedy bot beats the wall.
    var expanderTotal = 0
    var hedgehogTotal = 0
    for seat in 0 ..< Seats:
      if kinds[seat] == skExpander:
        expanderTotal += sim.centres(seat)
      else:
        hedgehogTotal += sim.centres(seat)
    check expanderTotal > hedgehogTotal

suite "baseline tuning":
  test "the tuning grid: every cell is legal and the shipped cell leads":
    ## tools/tune_baselines.nim in miniature, so the grid the expander's two
    ## numbers were chosen on is in every CI log rather than in someone's
    ## memory. The shipped cell is the design note's.
    let cells = sweep(@[1, 2], 2)
    check cells.len == 9
    var shipped = -1
    for cell in cells:
      echo cellLine(cell)
      check cell.complete
      check cell.illegal == 0
      if cell.tuning == DefaultExpanderTuning:
        shipped = cell.centres
        check cell.centres > cell.hedgehog
    check shipped >= 0

suite "no implicit offline fallback":
  test "decideAll without credentials fails model seats explicitly and still runs scripted seats":
    for key in ["COWORLD_LLM_ENDPOINT", "COWORLD_LLM_MODEL", "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY_URI", "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        "AWS_BEARER_TOKEN_BEDROCK"]:
      delEnv(key)
    let config = fixtureConfig(years = 2)
    var sim = initSim(config)
    let client = newLlmClient(config)
    check client.disabled
    let prompts = newSeq[string](Seats)
    let seats = sim.pendingSeats()
    ## Explicit scripted seats complete without credentials.
    let scriptedDecisions = client.decideAll(PromptSelection(), sim, seats,
      prompts, allOf(skExpander))
    check scriptedDecisions.len == Seats
    ## Non-scripted model seats fail explicitly instead of falling back.
    let modelKinds = newSeq[ScriptKind](Seats)
    expect DecisionFailure:
      discard client.decideAll(PromptSelection(), sim, seats, prompts, modelKinds)

suite "reply parsing":
  test "PLAYER_SCRIPTED accepts only canonical values":
    check parseScriptKind("expander") == skExpander
    check parseScriptKind(" EXPANDER ") == skExpander
    check parseScriptKind("hedgehog") == skHedgehog
    check parseScriptKind("") == skNone
    for bad in ["1", "true", "yes", "greedy", "turtle", "wall", "nonsense"]:
      expect CogplomacyError:
        discard parseScriptKind(bad)

  test "raw JSON and standard outer markdown JSON fence are accepted; prose is rejected":
    let raw = "{\"orders\":[\"A PAR - BUR\"]}"
    check extractJsonObject(raw){"orders"}.len == 1
    let fenced = "```json\n{\"orders\":[\"A PAR - BUR\"]}\n```"
    check extractJsonObject(fenced){"orders"}.len == 1
    let plainFenced = "```\n{\"orders\":[\"A PAR - BUR\"]}\n```"
    check extractJsonObject(plainFenced){"orders"}.len == 1
    let prose = "Here is my move.\n{\"orders\": [\"A PAR - BUR\"]}\nGood luck!"
    expect CogplomacyError:
      discard extractJsonObject(prose)
    expect CogplomacyError:
      discard extractJsonObject("no json at all here")

  test "invalid, illegal, or incomplete orders are rejected":
    var sim = initSim(fixtureConfig(years = 1, press = false))
    let seat = sim.seatOf[2] # France: A PAR, A MAR, F BRE
    expect CogplomacyError:
      discard sim.parseDecision(seat, parseJson("""{"notes":"hi"}"""))
    expect CogplomacyError:
      discard sim.parseDecision(seat, parseJson("""{"orders":"A PAR - BUR"}"""))
    ## Missing units are rejected instead of synthesized as holds.
    expect CogplomacyError:
      discard sim.parseDecision(seat, %*{"orders": ["A PAR - BUR"]})
    ## Illegal orders are rejected instead of repaired to holds.
    expect CogplomacyError:
      discard sim.parseDecision(seat,
        %*{"orders": ["A PAR - ENG", "A MAR H", "F BRE H"]})
    ## Redundant notation aliases are rejected.
    expect CogplomacyError:
      discard sim.parseDecision(seat,
        %*{"orders": ["A PAR -> BUR", "A MAR HOLD", "F BRE H"]})
    ## Canonical complete order set is accepted.
    let decision = sim.parseDecision(seat,
      %*{"orders": ["A PAR - BUR", "A MAR H", "F BRE H"]})
    check decision.orders == @["A PAR - BUR", "A MAR H", "F BRE H"]

  test "unknown pledge kind, alias, or unknown recipient is rejected":
    var sim = initSim(fixtureConfig(years = 1))
    let seat = sim.pendingSeats()[0]
    let otherPower = PowerNames[(sim.powerOf[seat] + 1) mod Powers]
    expect CogplomacyError:
      discard sim.parseDecision(seat, %*{
        "broadcast": "hello",
        "letters": [{"to": "ATLANTIS", "text": "hi"}]
      })
    expect CogplomacyError:
      discard sim.parseDecision(seat, %*{
        "broadcast": "hello",
        "pledges": [{"to": otherPower, "kind": "invade"}]
      })
    expect CogplomacyError:
      discard sim.parseDecision(seat, %*{
        "broadcast": "hello",
        "pledges": [{"to": "ALL", "kind": "keep-out", "province": "BUR"}]
      })
    expect CogplomacyError:
      discard sim.parseDecision(seat, %*{
        "broadcast": "hello",
        "pledges": [{"to": "ALL", "kind": "keepout", "province": "ZZZ"}]
      })
    let decision = sim.parseDecision(seat, %*{
      "broadcast": "hello",
      "letters": [{"to": otherPower, "text": "real"}],
      "pledges": [{"to": otherPower, "kind": "peace"},
                  {"to": "ALL", "kind": "keepout", "province": "BUR"}]
    })
    check decision.broadcast == "hello"
    check decision.letters.len == 1
    check decision.pledges.len == 2
    check decision.pledges[0].kind == plPeace
    check decision.pledges[1].kind == plKeepOut

  test "the error text for an unusable reply is cut on rune boundaries":
    try:
      discard extractJsonObject("🕊️".repeat(400))
      check false
    except CogplomacyError as error:
      check validateUtf8(error.msg) == -1
      check error.msg.runeLen <= 200

  test "a letter addressed to ALL is kept as a public letter":
    var sim = initSim(fixtureConfig(years = 1))
    let seat = sim.pendingSeats()[0]
    let decision = sim.parseDecision(seat, %*{
      "broadcast": "hello",
      "letters": [{"to": "all", "text": "everybody hears this"}]
    })
    check decision.letters.len == 1
    check decision.letters[0].toPower < 0
    check decision.letters[0].text == "everybody hears this"

  test "an over-long array is rejected":
    var sim = initSim(fixtureConfig(years = 1, press = false))
    let seat = sim.pendingSeats()[0]
    var many = newJArray()
    for index in 0 ..< 80:
      many.add(%"A PAR - BUR")
    expect CogplomacyError:
      discard sim.parseDecision(seat, %*{"orders": many})

suite "hosted movement prompts":
  test "London cannot support Norway and Naples cannot support Tunis":
    var game = initSim(fixtureConfig(seed = 7, press = false))
    # Keep London in place while Edinburgh moves to the North Sea.
    for seat in game.pendingSeats():
      var opening: seq[string]
      for unit in game.unitsOf(game.powerOf[seat]):
        opening.add(if unitLabel(unit) == "F EDI": "F EDI - NTH"
          else: unitLabel(unit) & " H")
      let decision = game.parseDecision(seat, %*{"orders": opening})
      game.applyOrders(seat, decision.orders, "", false)
    for (seat, illegal, legal) in [(1, "F LON S F NTH - NWY", "F LON S F NTH - ENG"),
        (3, "F NAP S A ROM - TUN", "F NAP S A ROM - APU")]:
      let prompt = game.userPrompt(seat, "")
      let view = game.observationJson(seat)
      check illegal notin prompt
      check legal in prompt
      let minimum = prompt.split("Minimum reply: ")[1].splitLines()[0]
      var reply = parseJson(minimum)
      discard game.parseDecision(seat, reply)
      for unit in view["units"]:
        for choice in unit["legalOrders"]:
          check not parseOrder(game.board, game.powerOf[seat], choice.getStr()).illegal
          check ("\n  " & choice.getStr() & "\n") in prompt
      var sourceIndex = -1
      for i, order in reply["orders"].elems:
        if order.getStr().startsWith(illegal[0 .. 4]): sourceIndex = i
      require sourceIndex >= 0
      reply["orders"].elems[sourceIndex] = %legal
      check legal in game.parseDecision(seat, reply).orders
      reply["orders"].elems[sourceIndex] = %illegal
      try:
        discard game.parseDecision(seat, reply)
        check false
      except CogplomacyError as error:
        check error.msg == "illegal order (nonadjacent): " & illegal
