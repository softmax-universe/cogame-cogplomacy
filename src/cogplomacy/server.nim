## Cogplomacy game server: implements the Coworld game contract.
##
## Endpoints:
##   GET /healthz                    - liveness
##   GET /client/global              - spectator page
##   GET /client/player              - player page (view-only; policies are prompts)
##   GET /client/replay              - replay page (replay mode)
##   GET /client/renderer.js         - shared stage renderer
##   GET /client/chrome.css          - shared chrome
##   GET /client/assets/<name>       - map, sprites and fonts
##   WS  /player?slot=N&token=T      - player protocol (prompt delivery)
##   WS  /global                     - spectator snapshots
##   WS  /replay                     - replay payload (replay mode)
##
## Player protocol (cogplomacy.player.v1), all JSON text frames:
##   game -> player: {"type":"welcome","slot":N,"power":...,"years":...}
##                   {"type":"state",...} after every event, redacted to
##                   what the seat may see (the whole board, but only its
##                   own press inbox)
##                   {"type":"final","scores":[...],"centres":[...]}
##   player -> game: {"type":"prompt","prompt":"...","scripted":"expander"}
##                   (max 4000 runes; scripted plays a built-in baseline for
##                   that seat: "expander" or "hedgehog"; "" means LLM-driven)

import
  std/[json, locks, os, sets, strutils, tables, times, unicode],
  bitworld/runtime,
  curly,
  mummy,
  mummy/routers,
  llm,
  prompts,
  sim,
  training

const
  MaxPromptLen = 4000
  ReplayVersion = 1
  ShutdownGraceSeconds = 20
    ## The certifier pings /global AFTER the player pods start, and a
    ## seven-baseline episode can be over in seconds. Keep answering for a
    ## bounded grace once the artifacts are written, then exit.

type
  GameState = object
    config: GameConfig
    selection: PromptSelection
    sim: Sim
    trajectory: DecisionTrajectory
    prompts: seq[string]
    scripted: seq[ScriptKind]
    promptSeen: seq[bool]
    slotErrors: seq[string]
    playerSockets: Table[int, WebSocket]
    socketSlots: Table[WebSocket, int]
    globalSockets: HashSet[WebSocket]
    started: bool
    finished: bool

var
  stateLock: Lock
  state: GameState
  gameServer: Server
  runtimeConfigGlobal: RuntimeConfig
  replayPayloadGlobal: string

initLock(stateLock)

proc clientDir(): string =
  let appDir = getAppDir()
  for candidate in [appDir / "client", appDir / ".." / "client", "client"]:
    if dirExists(candidate):
      return candidate
  "client"

proc dataDir(): string =
  let appDir = getAppDir()
  for candidate in [appDir / "data", appDir / ".." / "data", "data"]:
    if dirExists(candidate):
      return candidate
  "data"

proc policyNamesJson(gs: GameState): JsonNode =
  ## Seats play under anonymous cog aliases; the policy names ride alongside
  ## for the SPECTATOR views only, which render them in place of the aliases.
  result = newJArray()
  for player in gs.config.players:
    result.add(%player.name)

proc snapshotJson(gs: GameState): JsonNode =
  var events = newJArray()
  for event in gs.sim.events:
    events.add(event.eventToJson())
  var connected = newJArray()
  for slot in 0 ..< gs.config.tokens.len:
    connected.add(%gs.playerSockets.hasKey(slot))
  result = gs.sim.tableStateJson()
  result["type"] = %"state"
  result["game"] = %"cogplomacy"
  result["policyNames"] = gs.policyNamesJson()
  result["events"] = events
  result["started"] = %gs.started
  result["done"] = %gs.sim.done
  result["connected"] = connected

proc playerStateJson(gs: GameState, slot: int): JsonNode =
  result = gs.sim.observationJson(slot)
  result["started"] = %gs.started

proc broadcastLocked(gs: GameState) =
  ## Callers hold stateLock. Spectators get the whole table; players get the
  ## redacted per-seat state.
  let payload = $gs.snapshotJson()
  for socket in gs.globalSockets:
    socket.send(payload)
  for slot, socket in gs.playerSockets:
    socket.send($gs.playerStateJson(slot))

proc writeArtifact(uri, data, contentType, methodEnv: string) =
  ## Writes a Coworld artifact, honoring the platform's PUT/POST method hint.
  if uri.len == 0:
    return
  let httpMethod = getEnv(methodEnv, "PUT").toUpperAscii()
  if uri.isHttpCogameUri() and httpMethod == "POST":
    let curl = newCurly()
    var headers: HttpHeaders
    headers["content-type"] = contentType
    let response = curl.post(uri, headers, data, 60)
    if response.code < 200 or response.code >= 300:
      raise newException(IOError,
        "artifact POST failed: " & $response.code)
  else:
    writeCogameUri(uri, data, contentType, methodEnv)

proc replayPayload(gs: GameState, results: JsonNode): string =
  var names = newJArray()
  for name in gs.sim.names:
    names.add(%name)
  var powers = newJArray()
  for seat in 0 ..< Seats:
    powers.add(%PowerNames[gs.sim.powerOf[seat]])
  var events = newJArray()
  for event in gs.sim.events:
    events.add(event.eventToJson())
  $ %*{
    "protocol": "cogplomacy.replay.v" & $ReplayVersion,
    "names": names,
    "policyNames": gs.policyNamesJson(),
    "powers": powers,
    "config": {
      "years": gs.config.years,
      "seed": gs.config.seed,
      "press": gs.config.press,
      "sampled": true
    },
    "events": events,
    "results": results
  }

proc statesFromEvents(config: GameConfig, events: seq[GameEvent]): JsonNode =
  ## One table-state object per event prefix, for scrubbing replays.
  result = newJArray()
  for frame in replayMatch(config, events):
    result.add(frame.tableStateJson())

proc writePlayerFailure(failedSeat: int, reason: string) =
  ## Atomically publishes a typed GamePlayerFailure to COGAME_PLAYER_FAILURE_URI.
  let uri = getEnv("COGAME_PLAYER_FAILURE_URI").strip()
  if uri.len == 0:
    return
  let message = cleanText(oneLine(reason), 2000)
  let payload = $ %*{
    "message": (if message.len > 0: message else: "player decision failed"),
    "failed_policy_index": max(failedSeat, 0)
  }
  const FilePrefix = "file://"
  if uri.startsWith(FilePrefix):
    let path = uri[FilePrefix.len .. ^1]
    if path.len > 0:
      let dir = path.parentDir()
      if dir.len > 0:
        createDir(dir)
      let tmpPath = path & ".tmp." & $getCurrentProcessId()
      writeFile(tmpPath, payload)
      moveFile(tmpPath, path)
  else:
    writeArtifact(uri, payload, "application/json",
      "COGAME_PLAYER_FAILURE_METHOD")

proc failEpisode(failedSeat: int, reason: string) =
  ## Terminal explicit player failure: write GamePlayerFailure (never normal
  ## results/replay), preserve trajectory evidence if enabled, and exit nonzero.
  let cleanReason = cleanText(oneLine(reason), 2000)
  withLock stateLock:
    if state.finished:
      return
    state.finished = true
    state.trajectory.finishFailedTrajectory(failedSeat, cleanReason)
  echo "cogplomacy: failing episode (slot ", failedSeat, "): ", cleanReason
  writePlayerFailure(failedSeat, cleanReason)
  sleep(200)
  quit(1)

proc finishEpisode(runtimeConfig: RuntimeConfig) =
  var results: JsonNode
  var replayData: string
  withLock stateLock:
    if state.finished:
      return
    state.finished = true
    results = state.sim.resultsJson()
    replayData = state.replayPayload(results)
    state.trajectory.finishNativeTrajectory(results)

    ## Send final frames to players BEFORE writing artifacts: the hosted
    ## worker tears player pods down as soon as results.json exists, and
    ## writing first would race player log collection. Results carry POLICY
    ## names for the platform; the final frame carries the table ALIASES.
    var aliasNames = newJArray()
    for name in state.sim.names:
      aliasNames.add(%name)
    var final = %*{
      "type": "final",
      "done": true,
      "scores": results["scores"],
      "centres": results["centres"],
      "units": results["units"],
      "powers": results["powers"],
      "names": aliasNames,
      "years": results["years"],
      "reason": results["reason"],
      "soloist": results["soloist"]
    }
    for slot, socket in state.playerSockets:
      final["slot"] = %slot
      socket.send($final)
    state.broadcastLocked()

  sleep(500)
  echo "cogplomacy: writing results and replay"
  writeArtifact(
    runtimeConfig.resultsUri, $results, "application/json",
    "COGAME_RESULTS_METHOD"
  )
  writeArtifact(
    runtimeConfig.replayUri, replayData, "application/octet-stream",
    "COGAME_SAVE_REPLAY_METHOD"
  )
  echo "cogplomacy: artifacts written; answering health checks for ",
    ShutdownGraceSeconds, "s"
  sleep(ShutdownGraceSeconds * 1000)
  echo "cogplomacy: episode complete, shutting down"
  quit(0)

const PlayBudgetFraction* = 0.6
  ## Share of the platform's episode timeout spent playing. The rest covers
  ## container start, player connects, and writing the artifacts — the part
  ## that must never be the thing that runs out of time.

proc applyDecision(sim: var Sim, seat: int, decision: Decision,
    scripted: bool) =
  case sim.phase
  of pkPress:
    sim.applyPress(seat, decision.broadcast, decision.letters,
      decision.pledges, decision.notes, scripted)
  of pkOrders:
    sim.applyOrders(seat, decision.orders, decision.notes, scripted)
  of pkRetreats:
    sim.applyRetreats(seat, decision.retreats, decision.notes, scripted)
  of pkBuilds:
    sim.applyBuilds(seat, decision.adjustments, decision.notes, scripted)
  of pkDone:
    discard

proc runGame(runtimeConfig: RuntimeConfig) {.gcsafe.} =
  {.gcsafe.}:
    let config = state.config
    let gameStart = epochTime()
    let deadline = gameStart + config.playerConnectTimeoutSeconds

    while epochTime() < deadline:
      var allConnected = false
      withLock stateLock:
        allConnected = state.playerSockets.len >= config.tokens.len
      if allConnected:
        break
      sleep(200)

    ## Give the connected seats until the same connect deadline to deliver
    ## their prompt frames; a seat that never delivers a valid prompt fails the
    ## episode explicitly rather than silently falling back to a baseline.
    while epochTime() < deadline:
      var allDelivered = true
      var failedSlot = -1
      var failedReason = ""
      withLock stateLock:
        for slot in 0 ..< config.tokens.len:
          if state.slotErrors[slot].len > 0:
            failedSlot = slot
            failedReason = state.slotErrors[slot]
            break
          if not state.promptSeen[slot]:
            allDelivered = false
      if failedSlot >= 0:
        failEpisode(failedSlot, failedReason)
      if allDelivered:
        break
      sleep(100)

    var missingSlot = -1
    var missingReason = ""
    withLock stateLock:
      for slot in 0 ..< config.tokens.len:
        if state.slotErrors[slot].len > 0:
          missingSlot = slot
          missingReason = state.slotErrors[slot]
          break
        if not state.promptSeen[slot]:
          missingSlot = slot
          missingReason = "slot " & $slot & " did not deliver a prompt before connect timeout"
          break
      if missingSlot < 0:
        state.started = true
        echo "cogplomacy: starting with ", state.playerSockets.len, "/",
          config.tokens.len, " players connected"
        state.broadcastLocked()
    if missingSlot >= 0:
      failEpisode(missingSlot, missingReason)

    let client = newLlmClient(config)

    ## The platform kills the episode at its timeout and keeps nothing. Play
    ## inside a fraction of it so results and the replay are written with
    ## room to spare. The hosted dispatcher hands the timeout only to its own
    ## worker sidecar, NOT to the game container, so when the env is silent
    ## assume the configured platform default rather than playing open-ended.
    let hostedTimeout = getEnv("COWORLD_TIMEOUT_SECONDS", "").strip()
    var timeoutSeconds =
      if hostedTimeout.len > 0:
        try: parseFloat(hostedTimeout) except ValueError: 0.0
      else: 0.0
    if timeoutSeconds <= 0.0:
      timeoutSeconds = config.episodeTimeoutSeconds.float
    if timeoutSeconds <= 0.0:
      ## Play is never open-ended: a config carrying no usable timeout still
      ## gets the platform default the note names.
      timeoutSeconds = defaultGameConfig().episodeTimeoutSeconds.float
    let playDeadline = gameStart + timeoutSeconds * PlayBudgetFraction
    echo "cogplomacy: episode timeout ", timeoutSeconds.int, "s (",
      (if hostedTimeout.len > 0: "from env" else: "assumed"),
      "); playing until ", (timeoutSeconds * PlayBudgetFraction).int, "s"

    while true:
      var simCopy: Sim
      var seats: seq[int]
      var prompts: seq[string]
      var scripted: seq[ScriptKind]
      withLock stateLock:
        if state.sim.done:
          break
        ## Checked BEFORE every batch: past the budget we settle between
        ## phases and keep the episode rather than losing all of it.
        if epochTime() > playDeadline:
          echo "cogplomacy: episode deadline reached after ",
            state.sim.yearsPlayed, "/", config.years,
            " years; ending early"
          state.sim.endEarly()
          state.broadcastLocked()
          break
        seats = state.sim.pendingSeats()
        simCopy = state.sim
        prompts = state.prompts
        scripted = state.scripted
        echo "cogplomacy: ", state.sim.year, " ", state.sim.season, " ",
          state.sim.phase, " waiting on ", seats.len, " seats at ",
          (epochTime() - gameStart).int, "s"
      if seats.len == 0:
        break

      ## The slow part (Claude, ONE parallel batch for every live seat) runs
      ## outside the lock on a snapshot; only this thread mutates the sim, so
      ## the snapshot cannot go stale.
      var captureViews: seq[JsonNode]
      if not state.trajectory.isNil:
        for seat in seats: captureViews.add(simCopy.observationJson(seat))
      var decisions: seq[Decision]
      try:
        decisions = client.decideAll(state.selection, simCopy, seats, prompts, scripted)
      except DecisionFailure as error:
        if not state.trajectory.isNil:
          let pos = seats.find(error.seat)
          let view = if pos >= 0 and pos < captureViews.len: captureViews[pos]
            else: simCopy.observationJson(error.seat)
          let attempts = if error.seat >= 0 and error.seat < client.nativeAttempts.len:
            client.nativeAttempts[error.seat] else: @[]
          state.trajectory.recordFailedDecision(simCopy.events.len, error.seat,
            config.players[error.seat].name, view, attempts, state.selection, error.msg)
        failEpisode(error.seat, error.msg)

      var applyFailedSeat = -1
      var applyFailedReason = ""
      withLock stateLock:
        for index, seat in seats:
          let wasScripted = scripted[seat] != skNone
          let eventIndex = state.sim.events.len
          try:
            applyDecision(state.sim, seat, decisions[index], wasScripted)
          except CatchableError as error:
            applyFailedSeat = seat
            applyFailedReason = "reply rejected: " & error.msg
            if not state.trajectory.isNil:
              let attempts = if seat >= 0 and seat < client.nativeAttempts.len:
                client.nativeAttempts[seat] else: @[]
              state.trajectory.recordFailedDecision(eventIndex, seat,
                config.players[seat].name, captureViews[index], attempts,
                state.selection, applyFailedReason)
            break
          if not state.trajectory.isNil:
            let applied = state.sim.events[eventIndex]
            state.trajectory.recordAppliedDecision(eventIndex, seat,
              config.players[seat].name, captureViews[index], applied.executedReply(),
              applied.scripted, scripted[seat] != skNone, client.nativeAttempts[seat],
              state.selection)
        if applyFailedSeat < 0:
          state.broadcastLocked()
      if applyFailedSeat >= 0:
        failEpisode(applyFailedSeat, applyFailedReason)

      ## Pace between phases so spectators can read the board.
      if config.turnDelayMs > 0:
        sleep(config.turnDelayMs)

    if config.turnDelayMs > 0:
      sleep(config.turnDelayMs)
    finishEpisode(runtimeConfig)

var gameThread: Thread[RuntimeConfig]

proc serveFile(request: Request, path, contentType: string) =
  if fileExists(path):
    var headers: HttpHeaders
    headers["Content-Type"] = contentType
    request.respond(200, headers, readFile(path))
  else:
    request.respond(404)

proc htmlHandler(name: string): RequestHandler =
  proc handler(request: Request) {.gcsafe.} =
    {.gcsafe.}:
      serveFile(request, clientDir() / name, "text/html; charset=utf-8")
  handler

proc assetHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let name = request.pathParams["name"]
    if "/" in name or "\\" in name or name.startsWith("."):
      request.respond(404)
      return
    let contentType =
      if name.endsWith(".png"): "image/png"
      elif name.endsWith(".ttf"): "font/ttf"
      elif name.endsWith(".json"): "application/json"
      else: "application/octet-stream"
    serveFile(request, dataDir() / name, contentType)

proc rendererHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    serveFile(
      request, clientDir() / "renderer.js",
      "application/javascript; charset=utf-8"
    )

proc chromeCssHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    serveFile(
      request, clientDir() / "chrome.css",
      "text/css; charset=utf-8"
    )

proc healthzHandler(request: Request) {.gcsafe.} =
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, """{"ok": true}""")

proc playerUpgradeHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let slotText = request.queryParams["slot"]
    let token = request.queryParams["token"]
    var slot = -1
    try:
      slot = parseInt(slotText)
    except ValueError:
      discard
    var authorized = false
    withLock stateLock:
      authorized = slot >= 0 and slot < state.config.tokens.len and
        state.config.tokens[slot] == token
    if not authorized:
      request.respond(401)
      return
    let websocket = request.upgradeToWebSocket()
    withLock stateLock:
      state.playerSockets[slot] = websocket
      state.socketSlots[websocket] = slot
      echo "cogplomacy: player slot ", slot, " connected (",
        state.playerSockets.len, "/", state.config.tokens.len, ")"
      websocket.send($ %*{
        "type": "welcome",
        "protocol": "cogplomacy.player.v1",
        "slot": slot,
        "power": PowerNames[state.sim.powerOf[slot]],
        "years": state.config.years,
        "press": state.config.press
      })

proc globalUpgradeHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let websocket = request.upgradeToWebSocket()
    withLock stateLock:
      state.globalSockets.incl(websocket)
      websocket.send($state.snapshotJson())

proc replayUpgradeHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let websocket = request.upgradeToWebSocket()
    if replayPayloadGlobal.len > 0:
      websocket.send(replayPayloadGlobal)

proc websocketHandler(
  websocket: WebSocket,
  event: WebSocketEvent,
  message: Message
) {.gcsafe.} =
  {.gcsafe.}:
    case event
    of OpenEvent:
      discard
    of MessageEvent:
      ## mummy hands Ping frames to the application instead of answering
      ## them itself; the platform's certifier pings /global to check the
      ## game is alive, so an unanswered ping fails certification.
      if message.kind == Ping:
        websocket.send(message.data, Pong)
        return
      if message.kind != TextMessage:
        return
      var slot = -1
      withLock stateLock:
        slot = state.socketSlots.getOrDefault(websocket, -1)
      if slot < 0:
        return
      try:
        let payload = parseJson(message.data)
        if payload{"type"}.getStr() == "prompt":
          var prompt = payload{"prompt"}.getStr()
          if prompt.runeLen > MaxPromptLen:
            prompt = prompt.runeSubStr(0, MaxPromptLen)
          let node = payload{"scripted"}
          let scripted =
            if node.isNil: skNone
            elif node.kind == JString: parseScriptKind(node.getStr())
            else:
              raise newException(CogplomacyError, "scripted field must be a string")
          withLock stateLock:
            if state.started and not state.promptSeen[slot]:
              echo "cogplomacy: slot ", slot,
                " delivered a prompt after play started; ignoring it"
            else:
              state.prompts[slot] = prompt
              state.scripted[slot] = scripted
              state.promptSeen[slot] = true
              echo "cogplomacy: slot ", slot, " delivered a prompt (",
                prompt.len, " chars",
                (if scripted != skNone: ", scripted " & $scripted else: ""),
                ")"
      except CatchableError as error:
        echo "cogplomacy: rejecting bad player frame from slot ", slot, ": ", error.msg
        withLock stateLock:
          if not state.promptSeen[slot] and state.slotErrors[slot].len == 0:
            state.slotErrors[slot] = "slot " & $slot & " sent invalid registration frame: " & error.msg
    of ErrorEvent:
      discard
    of CloseEvent:
      withLock stateLock:
        if websocket in state.socketSlots:
          let slot = state.socketSlots[websocket]
          state.socketSlots.del(websocket)
          if state.playerSockets.getOrDefault(slot) == websocket:
            state.playerSockets.del(slot)
        state.globalSockets.excl(websocket)

proc buildRouter(replayMode: bool): Router =
  result.get("/healthz", healthzHandler)
  result.get("/client/global", htmlHandler("global.html"))
  result.get("/client/player", htmlHandler("player.html"))
  result.get("/client/replay", htmlHandler("replay.html"))
  result.get("/client/renderer.js", rendererHandler)
  result.get("/client/chrome.css", chromeCssHandler)
  result.get("/client/assets/@name", assetHandler)
  result.get("/global", globalUpgradeHandler)
  result.get("/replay", replayUpgradeHandler)
  if not replayMode:
    result.get("/player", playerUpgradeHandler)

proc configFromReplay*(payload: JsonNode): GameConfig =
  result = defaultGameConfig()
  result.years = payload["config"]{"years"}.getInt(4)
  result.seed = payload["config"]{"seed"}.getInt(0)
  result.press = payload["config"]{"press"}.getBool(true)
  ## The replay carries the episode's fitted cap; never re-fit it. The
  ## seat→power permutation and the aliases are re-derived from the seed.
  result.sampled = true
  for name in payload["names"]:
    result.players.add(PlayerConfig(name: name.getStr()))

proc runReplayServer*(runtimeConfig: RuntimeConfig) =
  ## Replay mode: parse the recorded replay, precompute the scrub states,
  ## and serve the viewer until the platform tears the container down.
  let payload = parseJson(runtimeConfig.replay)
  let config = configFromReplay(payload)
  var events: seq[GameEvent]
  for node in payload["events"]:
    events.add(eventFromJson(node))
  var enriched = %*{
    "type": "replay",
    "protocol": payload{"protocol"}.getStr("cogplomacy.replay.v1"),
    "names": payload["names"],
    "policyNames": payload{"policyNames"},
    "powers": payload{"powers"},
    "config": payload["config"],
    "events": payload["events"],
    "results": payload{"results"},
    "states": statesFromEvents(config, events)
  }
  replayPayloadGlobal = $enriched

  let router = buildRouter(replayMode = true)
  gameServer = newServer(router, websocketHandler)
  echo "cogplomacy: replay mode on ", runtimeConfig.host, ":",
    runtimeConfig.port
  gameServer.serve(Port(runtimeConfig.port), runtimeConfig.host)

proc runGameServer*(config: GameConfig, runtimeConfig: RuntimeConfig) =
  if config.tokens.len != config.players.len:
    raise newException(CogplomacyError, "tokens and players must align")
  state.config = config
  state.selection = promptSelection(parseJson(
    if runtimeConfig.config.len > 0: runtimeConfig.config else: "{}"))
  state.sim = initSim(config)
  state.trajectory = initNativeTrajectory(config)
  state.prompts = newSeq[string](config.players.len)
  state.scripted = newSeq[ScriptKind](config.players.len)
  state.promptSeen = newSeq[bool](config.players.len)
  state.slotErrors = newSeq[string](config.players.len)
  runtimeConfigGlobal = runtimeConfig

  let router = buildRouter(replayMode = false)
  gameServer = newServer(router, websocketHandler)
  createThread(gameThread, runGame, runtimeConfig)
  echo "cogplomacy: serving on ", runtimeConfig.host, ":", runtimeConfig.port
  gameServer.serve(Port(runtimeConfig.port), runtimeConfig.host)
