## Hosted calls must reach the native sidecar without provider credentials.
include "../src/cogplomacy/llm"

block:
  putEnv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:9100/")
  putEnv("COWORLD_LLM_MODEL", "anthropic/claude-sonnet-4.6")
  putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://retired.invalid")
  putEnv("ANTHROPIC_API_KEY", "local-key-must-not-be-used")
  let client = newLlmClient(GameConfig())
  for slot in 0 .. 1:
    let request = client.requestFor("rules", "private view", slot)
    doAssert request.url == "http://127.0.0.1:9100/v1/messages"
    doAssert request.headers["X-Coworld-Player-Slot"] == $slot
    let body = parseJson(request.body)
    doAssert body["model"].getStr() == "anthropic/claude-sonnet-4.6"
    doAssert not body.hasKey("anthropic_version")
    doAssert not body.hasKey("output_config")
  echo "hosted sidecar routing and seat attribution passed"

block:
  doAssert failureKind(Response(code: 503), "") == fkInconclusive
  doAssert failureKind(Response(), "timeout") == fkInconclusive
  doAssert failureKind(Response(code: 429, body: "temporarily throttled"), "") == fkInconclusive
  doAssert failureKind(Response(code: 429, body:
    """{"error":{"message":"episode LLM spend limit reached ($0.10)"}}"""), "") == fkPlayerError
  doAssert failureKind(Response(code: 200), "") == fkPlayerError

  for body in ["[]", "{}", """{"content":"bad"}"""]:
    let response = Response(code: 200, body: body)
    var attempt = nativeAttempt("model", 0, 0, 0, """{"system":"","messages":[]}""")
    attempt.receiveNativeResponse(response, "", 0)
    doAssert attempt.rawResponse.getStr() == body
    try:
      discard textOf(LlmClient(), response, "", "fixture")
      doAssert false
    except CogplomacyError as error:
      doAssert error.msg == "llm response requires a content array"
