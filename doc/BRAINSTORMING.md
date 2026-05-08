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
 * It is OK to use "modern" python (3.12) and its features (types!)
 * Out of personal interest, I try to do everything with local models and tools where possible and try to not use openai etc.
 * Also due to personal interest I opted for chainlit because I really want to try it out - sorry ;-)
 * in general, I did take the opportunity to try out some things... I hope that is ok (e.g. pyright strict mode, uv,...)
 * It is OK to hardcode a certain provider/model
 * There are no special privacy requirements (e.g. using hosted llm is fine)
 * This is not about client/server vs desktop/local app, I can choose freely (will probably choose local cli app)
 * There are no special UX or Auth requirements, it is fine to do a username/password "inline" to identify the user for our external service - I think that env variables would be not very realistic. Of course one could also do a global login - in reality we might want a global auth against a SSO service, but I thought that the two-step tool call might be more interesting and that there might be real usecases for this, too (external systems that do not integrate with SSO).
 * The auth for the vacation days service does not properly hide the password - in a real implementation we would have to discuss other options, but I understood that the UI is no focus anyways...
 

---
Decisions
 * I am intentionally not following YAGNI strictly in places that I expect to be volatile, I already use clean structures (e.g. model dependent tool descriptions, configurable provider/model etc.) where I expect it to be necessary in the longer run, regardless of current need.
 * Multimodal is achieved with document content extraction approach instead of shared embeddings. Reason: it is said to be more reliable
 * To show how I work realistically, I did work a lot with protocols and decoupling as well as manual constructor based IOC/DI
 * I am relying on ollama and local models
 * RAG as a Tool - the extra LLM call should be fine with local model and the (hopefully improved) quality should be worth it
 * For real projects, we would probably use uv workspaces and have separate chatbot and ingest packages. For simplicity, I use toplevel modules instead here.

--- 
Future Improvements
 * Implement document upload
 * Convert into monorepo and use uv workspace 
 * Think about client/server, maybe server as docker image
 * Maybe MCP support (just for the fun of it :-))
 * Probably improve auth by integrating actually realistic SSO or such - gradio apparently even supports this natively
 * Red Teaming my own project
 * further evaluate and improve accuracy
 * automated tests / CI Pipeline for evaluations (CircleCI)
 

---
Tech Stack:

* chainlit for UI
* First version: lets see how far we can go with privacy-friendly local solutions
  * haystack with LLM Document Content Extractor for multimodal
  * qdrant (faster than chromadb)
  * ollama
  * Llama-3.2-Vision
  * Llama-3.1

--- 
Soft Requirements:
 * README shall be well documented, shall contain steps like the instruction on how to start the qdrant docker container etc., to make it easy to play with this.

---
 Steps:

 1. pyenv or what to use best nowadays for managing python installation, dependency management etc?
 2. basic chatbot that talks to some LLM
 3. add toolcalling for external service call (simulate an external "vacation days left" service) with pydantic
 4. add RAG ingestion of text documents  (txt, markdown)
 5. wire up the RAG with the chatbot
 6. add RAG ingestion of pdf documents
