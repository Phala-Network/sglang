//! CPU boundary regression: use the production protocol type, serde conversion
//! and reqwest JSON request body, with local P/D mock workers. No GPU/model.
use openai_protocol::chat::ChatCompletionRequest;
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::Arc,
    thread,
};

fn roundtrip(include: Option<bool>, stream: bool) {
    let mut input = json!({"model":"deepseek/deepseek-v4-flash-0731",
        "messages":[{"role":"user","content":"What is 11 + 12?"}],
        "temperature":0,"max_tokens":1024,"n":2,
        "chat_template_kwargs":{"thinking":true},"stream":stream});
    if let Some(value) = include {
        input["include_reasoning"] = json!(value);
    }
    // Same typed Arc -> serde_json::to_value boundary used by pd_router.rs,
    // and the same protocol type used by the regular OpenAI router.
    let request: Arc<ChatCompletionRequest> = Arc::new(serde_json::from_value(input).unwrap());
    let forwarded = serde_json::to_value(request.as_ref()).unwrap();
    assert_eq!(
        forwarded.get("include_reasoning"),
        include.as_ref().map(|v| if *v {
            &Value::Bool(true)
        } else {
            &Value::Bool(false)
        }),
        "typed serde dropped include_reasoning"
    );
    assert_eq!(forwarded["n"], 2);
    assert_eq!(forwarded["chat_template_kwargs"]["thinking"], true);
    assert_eq!(forwarded["stream"], stream);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let expected = forwarded.clone();
    let response_body = if stream {
        "data: {\"choices\":[{\"delta\":{\"content\":\"23\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":21,\"reasoning_tokens\":18}}\n\ndata: [DONE]\n\n".to_owned()
    } else {
        json!({"choices":[{"message":{"content":"23"},"finish_reason":"stop"}],"usage":{"completion_tokens":21,"reasoning_tokens":18,"completion_tokens_details":{"reasoning_tokens":18}}}).to_string()
    };
    let expected_response = response_body.clone();
    let server = thread::spawn(move || {
        for _ in 0..2 {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(10)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0u8; 4096];
            let header_end = loop {
                let n = socket.read(&mut buffer).unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buffer[..n]);
                if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    break end + 4;
                }
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|s| s.trim().parse().unwrap())
                })
                .unwrap();
            while bytes.len() < header_end + length {
                let n = socket.read(&mut buffer).unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buffer[..n]);
            }
            let body: Value =
                serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap();
            assert_eq!(
                body, expected,
                "worker must receive the typed request without field loss"
            );
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n{}",response_body.len(),if stream {"text/event-stream"} else {"application/json"},response_body).unwrap();
        }
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();
    // Both PD destinations receive JSON via reqwest, just as the production
    // build_post_with_headers path. This does not instantiate the whole router.
    for worker in ["prefill", "decode"] {
        let response = client
            .post(format!("http://{address}/{worker}/v1/chat/completions"))
            .json(&forwarded)
            .send()
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.text().unwrap(),
            expected_response,
            "response and usage bytes must be preserved"
        );
    }
    server.join().unwrap();
}
#[test]
fn false_nonstream() {
    roundtrip(Some(false), false)
}
#[test]
fn true_nonstream() {
    roundtrip(Some(true), false)
}
#[test]
fn default_nonstream() {
    roundtrip(None, false)
}
#[test]
fn false_stream() {
    roundtrip(Some(false), true)
}
#[test]
fn true_stream() {
    roundtrip(Some(true), true)
}
#[test]
fn default_stream() {
    roundtrip(None, true)
}
#[test]
fn invalid_boolean_is_rejected() {
    for value in [json!("false"), json!(0)] {
        let request = json!({"model":"fixture","messages":[],"include_reasoning":value});
        assert!(serde_json::from_value::<ChatCompletionRequest>(request).is_err());
    }
}
