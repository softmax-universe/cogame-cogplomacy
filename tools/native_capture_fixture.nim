## Full native server fixture: each response is precomputed by the production kernel.
import std/[json, os, strutils]
import cogplomacy/[llm, prompts]

proc apply(sim: var Sim, seat: int, d: Decision, scripted: bool) =
  case sim.phase
  of pkPress: sim.applyPress(seat, d.broadcast, d.letters, d.pledges, d.notes, scripted)
  of pkOrders: sim.applyOrders(seat, d.orders, d.notes, scripted)
  of pkRetreats: sim.applyRetreats(seat, d.retreats, d.notes, scripted)
  of pkBuilds: sim.applyBuilds(seat, d.adjustments, d.notes, scripted)
  of pkDone: doAssert false

let args = commandLineParams()
doAssert args.len == 2
var config = defaultGameConfig()
config.update(readFile(args[0]))
config = config.sampleEpisode()
var game = initSim(config)
let selection = promptSelection(newJObject())
var requests = newJArray()
var expected = newJArray()
var phases = newJObject()
var phaseIndex = 0
while not game.done:
  let seats = game.pendingSeats()
  let phase = game.phase
  phases[$phase] = %(phases{$phase}.getInt() + 1)
  var views: array[Seats, JsonNode]
  var decisions: array[Seats, Decision]
  for seat in seats:
    views[seat] = game.observationJson(seat)
    let system = systemPrompt(game, seat, selection)
    let user = userPrompt(game, seat, "PRIVATE_OPERATOR_" & $seat)
    let minimum = parseJson(user.split("Minimum reply: ")[1].splitLines()[0])
    discard game.parseDecision(seat, minimum)
    var proposal = scriptedDecision(game, seat, skExpander).decisionReply(phase)
    proposal["notes"] = %("PRIVATE_CAPTURE_" & $seat)
    if phase == pkPress:
      proposal["broadcast"] = %("PUBLIC_" & $game.year & "_" & $game.season & "_" & $seat)
      proposal["letters"] = %*[{"to": PowerNames[(game.powerOf[seat] + 1) mod Seats],
        "text": "PRIVATE_LETTER_" & $seat}]
    if phaseIndex == 2: proposal["notes"] = %("é".repeat(5000))
    if phaseIndex == 4: proposal.delete("notes")
    let text =
      if phaseIndex == 3: "```json\n" & $proposal & "\n```"
      else: $proposal
    if phaseIndex == 1:
      let invalidText = "invalid fixture reply"
      let errMsg = "expected JSON object in response: " & invalidText
      requests.add(%*{"slot": seat, "system": system, "user": user, "response": invalidText})
      requests.add(%*{"slot": seat, "system": system,
        "user": user & "\n\nYour previous reply was rejected (" & errMsg &
          "). Respond with ONLY the requested valid JSON object.",
        "response": text})
    else:
      requests.add(%*{"slot": seat, "system": system, "user": user, "response": text})
    decisions[seat] = game.parseDecision(seat, proposal)
  for seat in seats:
    let eventIndex = game.events.len
    game.apply(seat, decisions[seat], false)
    let event = game.events[eventIndex]
    doAssert event.kind in {evPress, evOrders, evRetreat, evBuild}
    expected.add(%*{"event_index": eventIndex, "seat": seat, "observation": views[seat],
      "action": event.executedReply(), "scripted": event.scripted, "phase": $phase,
      "phase_index": phaseIndex})
  inc phaseIndex
var events = newJArray()
for event in game.events: events.add(event.eventToJson())
doAssert game.reason == "complete" or game.reason == "solo"
writeFile(args[1], pretty(%*{"requests": requests, "expected": expected,
  "events": events, "phases": phases, "results": game.resultsJson()}) & "\n")
echo "complete fixture phases: ", phases
