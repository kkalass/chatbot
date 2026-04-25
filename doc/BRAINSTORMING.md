* sounds like RAG & LLM 
* UI ideas
  * just a telegram bot maybe? => no e2ee, not trustworthy, unsuited for potentially sensitive data. Is this an issue here? Maybe
  * a matrix bot => secure, maybe a bit more involved, but actually something I would like to try (matrix-nio is the python lib). But maybe too much for this
  * cli => cli is probably better suited than telegram or matrix here, maybe with python `rich` package or `prompt_toolkit`
  * web app => `gradio` is sort of standard for those things, maybe easiest option. Or `streamlit` or even `chainlit`?
* External Service: The example implies authentication/authorization
* 

* Basic architecture decisions: 
  * There was no statement about client/server or desktop/local app. I guess the 

---
Research Topics
 * What is the easiest UI option for start? => probably gradio (with simple auth: `gr.ChatInterface(...).launch(auth=("user", "password"))`)
 * MCP for tool calls, or is this absolute overkill? => yes, overkill
 * If MCP: What are the best practices regarding auth nowadays ("You know: the S in MCP stands for security...")
 * Or rather specific, hardcoded tool (external service), maybe with pydantic model? => yes
 * what RAG? How to do the data ingestion etc
 * what about monitoring? Especially evaluation (e.g. quality monitoring)?
 * [what are best practices for automated testing of such a tool]

---
Assumptions
 * It is OK to use RAG/LLM services
 * It is OK to hardcode a certain provider/model
 * There are no special privacy requirements (e.g. using hosted llm is fine)
 * This is not about client/server vs desktop/local app, I can choose freely (will probably choose local cli app)
 * There are no special UX or Auth requirements, it is fine to do a username/password to identify the user for our external service


--- 
Future Improvements
 * Implement document upload
 * Make model switchable
 * Think about client/server, maybe server as docker image
 * Maybe MCP support (just for the fun of it :-))
 * Probably improve auth by integrating actually realistic SSO or such - gradio apparently even supports this natively
 * Use Chainlint instead of Gradio (less simple, but so much more impressive)
 * Red Teaming my own project
 * evaluate and improve accuracy
 * automated tests / CI Pipeline for evaluations (CircleCI)
 * implement advanced RAG retrieval methods