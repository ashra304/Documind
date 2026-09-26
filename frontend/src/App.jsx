import { useEffect, useState } from "react"
import "./App.css"

const API_URL = "http://127.0.0.1:8000"

function App() {
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  const [sessionId, setSessionId] = useState(
    () => crypto.randomUUID()
  )

  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState("")

  const [documents, setDocuments] = useState([])
  const [activeDocument, setActiveDocument] = useState(null)

  // --------------------------------------------------
  // Load documents
  // --------------------------------------------------

  const fetchDocuments = async () => {
  try {
    const response = await fetch(
      `${API_URL}/documents`
    )

    if (!response.ok) {
      throw new Error(
        "Failed to load documents"
      )
    }

    const data = await response.json()

    setDocuments(
      data.documents || []
    )
  } catch (error) {
    console.error(
      "Failed to load documents:",
      error
    )
  }
}

useEffect(() => {
  let cancelled = false

  const loadDocuments = async () => {
    try {
      const response = await fetch(
        `${API_URL}/documents`
      )

      if (!response.ok) {
        throw new Error(
          "Failed to load documents"
        )
      }

      const data =
        await response.json()

      if (!cancelled) {
        setDocuments(
          data.documents || []
        )
      }

    } catch (error) {
      if (!cancelled) {
        console.error(
          "Failed to load documents:",
          error
        )
      }
    }
  }

  loadDocuments()

  return () => {
    cancelled = true
  }
}, [])

  // --------------------------------------------------
  // New chat
  // --------------------------------------------------

  const newChat = () => {
    setMessages([])
    setQuestion("")
    setSessionId(crypto.randomUUID())
  }

  // --------------------------------------------------
  // Upload document
  // --------------------------------------------------

  const uploadDocument = async () => {
    if (!selectedFile) {
      setUploadMessage(
        "Please select a PDF first."
      )
      return
    }

    setUploading(true)
    setUploadMessage("Uploading and indexing...")

    const formData = new FormData()

    formData.append(
      "file",
      selectedFile
    )

    try {
      const response = await fetch(
        `${API_URL}/upload`,
        {
          method: "POST",
          body: formData
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Upload failed."
        )
      }

      setUploadMessage(
        `✓ ${data.filename} uploaded successfully`
      )

      setSelectedFile(null)

      await fetchDocuments()

      setActiveDocument(
        data.filename
      )

      setMessages([])
      setSessionId(
        crypto.randomUUID()
      )

    } catch (error) {
      console.error(error)

      setUploadMessage(
        error.message ||
        "Could not upload the document."
      )
    } finally {
      setUploading(false)
    }
  }

  // --------------------------------------------------
  // Delete document
  // --------------------------------------------------

  const deleteDocument = async (
    filename
  ) => {
    const confirmed = window.confirm(
      `Delete "${filename}"?`
    )

    if (!confirmed) {
      return
    }

    try {
      const response = await fetch(
        `${API_URL}/documents/${encodeURIComponent(
          filename
        )}`,
        {
          method: "DELETE"
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Delete failed."
        )
      }

      if (
        activeDocument === filename
      ) {
        setActiveDocument(null)
        setMessages([])
        setSessionId(
          crypto.randomUUID()
        )
      }

      await fetchDocuments()

    } catch (error) {
      console.error(error)

      alert(
        error.message ||
        "Could not delete document."
      )
    }
  }

  // --------------------------------------------------
  // Select document
  // --------------------------------------------------

  const selectDocument = (
    filename
  ) => {
    setActiveDocument(filename)
    setMessages([])
    setQuestion("")
    setSessionId(
      crypto.randomUUID()
    )
  }

  // --------------------------------------------------
  // Summarize document
  // --------------------------------------------------

  const summarizeDocument = async () => {
    if (!activeDocument || loading) {
      return
    }

    setLoading(true)

    try {
      const response = await fetch(
        `${API_URL}/summarize`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json"
          },
          body: JSON.stringify({
            document_name:
              activeDocument
          })
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Summarization failed."
        )
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.summary,
          sources: []
        }
      ])

    } catch (error) {
      console.error(error)

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Sorry, I could not summarize this document.",
          sources: []
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  // --------------------------------------------------
  // Ask question
  // --------------------------------------------------

  const askQuestion = async () => {
    const currentQuestion =
      question.trim()

    if (
      !currentQuestion ||
      loading
    ) {
      return
    }

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: currentQuestion
      }
    ])

    setQuestion("")
    setLoading(true)

    try {
      const response = await fetch(
        `${API_URL}/query`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json"
          },
          body: JSON.stringify({
            question:
              currentQuestion,
            session_id:
              sessionId,
            document_name:
              activeDocument
          })
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Something went wrong."
        )
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            data.answer ||
            "No answer returned.",
          sources:
            data.sources || []
        }
      ])

    } catch (error) {
      console.error(error)

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Sorry, I could not connect to the DocuMind backend.",
          sources: []
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  // --------------------------------------------------
  // Enter key
  // --------------------------------------------------

  const handleKeyDown = (
    event
  ) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault()
      askQuestion()
    }
  }

  // --------------------------------------------------
  // Render
  // --------------------------------------------------

  return (
    <div className="app">

      {/* Header */}

      <header className="header">

        <div>
          <h1>DocuMind</h1>

          <p>
            Multi-Document RAG Intelligence Platform
          </p>
        </div>

        <button
          className="new-chat-button"
          onClick={newChat}
        >
          + New Chat
        </button>

      </header>


      {/* Main layout */}

      <div className="main-layout">

        {/* Sidebar */}

        <aside className="sidebar">

          <div className="sidebar-header">

            <div>
              <h2>Documents</h2>

              <span>
                {documents.length} documents
              </span>
            </div>

          </div>


          {/* Upload */}

          <div className="upload-card">

            <h3>
              Upload PDF
            </h3>

            <p>
              Add a document to your knowledge base.
            </p>

            <input
              type="file"
              accept=".pdf"
              onChange={(event) =>
                setSelectedFile(
                  event.target.files[0]
                )
              }
            />

            {selectedFile && (
              <div className="selected-file">
                {selectedFile.name}
              </div>
            )}

            <button
              className="upload-button"
              onClick={uploadDocument}
              disabled={
                uploading ||
                !selectedFile
              }
            >
              {uploading
                ? "Indexing..."
                : "Upload Document"}
            </button>

            {uploadMessage && (
              <div className="upload-message">
                {uploadMessage}
              </div>
            )}

          </div>


          {/* Document list */}

          <div className="document-list">

            {documents.length === 0 && (
              <div className="empty-documents">
                No documents yet.
              </div>
            )}

            {documents.map(
              (document) => {

                const filename =
                  document.filename

                const isActive =
                  activeDocument ===
                  filename

                return (
                  <div
                    key={filename}
                    className={
                      isActive
                        ? "document-row active"
                        : "document-row"
                    }
                  >

                    <button
                      className="document-item"
                      onClick={() =>
                        selectDocument(
                          filename
                        )
                      }
                    >

                      <span className="document-icon">
                        📄
                      </span>

                      <span className="document-name">
                        {filename}
                      </span>

                    </button>

                    <button
                      className="delete-document"
                      onClick={() =>
                        deleteDocument(
                          filename
                        )
                      }
                      title="Delete document"
                    >
                      ×
                    </button>

                  </div>
                )
              }
            )}

          </div>

        </aside>


        {/* Chat area */}

        <main className="chat-container">

          {/* Chat header */}

          <div className="chat-header">

            <div>

              <span className="status-dot"></span>

              <span>
                DocuMind AI
              </span>

            </div>

            {activeDocument && (
              <span className="active-document-badge">
                📄 {activeDocument}
              </span>
            )}

          </div>


          {/* Messages */}

          <div className="messages">

            {messages.length === 0 && (
              <div className="welcome">

                <div className="welcome-icon">
                  ✦
                </div>

                <h2>
                  Ask your documents anything.
                </h2>

                <p>
                  DocuMind retrieves relevant
                  information from your documents
                  and generates grounded answers.
                </p>

                <div className="welcome-hints">

                  <button
                    onClick={() =>
                      setQuestion(
                        "What are the main topics discussed in this document?"
                      )
                    }
                  >
                    Summarize a document
                  </button>

                  <button
                    onClick={() =>
                      setQuestion(
                        "What are the key concepts in this document?"
                      )
                    }
                  >
                    Find key concepts
                  </button>

                </div>

              </div>
            )}


            {messages.map(
              (message, index) => (

                <div
                  key={index}
                  className={
                    `message ${message.role}`
                  }
                >

                  <div className="message-label">

                    {message.role ===
                    "user"
                      ? "You"
                      : "DocuMind"}

                  </div>

                  <div className="message-content">
                    {message.content}
                  </div>


                  {/* Sources */}

                  {message.role ===
                    "assistant" &&
                    message.sources &&
                    message.sources.length >
                      0 && (

                    <div className="sources">

                      <div className="sources-title">
                        Sources
                      </div>

                      {message.sources.map(
                        (
                          source,
                          sourceIndex
                        ) => (

                          <div
                            className="source"
                            key={
                              sourceIndex
                            }
                          >

                            <span>
                              📄{" "}
                              {
                                source.document
                              }
                            </span>

                            <span>
                              Page{" "}
                              {
                                source.page
                              }
                            </span>

                          </div>

                        )
                      )}

                    </div>

                  )}

                </div>

              )
            )}


            {/* Loading */}

            {loading && (
              <div className="message assistant">

                <div className="message-label">
                  DocuMind
                </div>

                <div className="message-content thinking">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>

              </div>
            )}

          </div>


          {/* Active document controls */}

          {activeDocument && (
            <div className="document-actions">

              <button
                onClick={
                  summarizeDocument
                }
                disabled={loading}
              >
                📝 Summarize Document
              </button>

            </div>
          )}


          {/* Input */}

          <div className="input-area">

            <input
              type="text"
              placeholder={
                activeDocument
                  ? `Ask about ${activeDocument}...`
                  : "Ask anything about your documents..."
              }
              value={question}
              onChange={(event) =>
                setQuestion(
                  event.target.value
                )
              }
              onKeyDown={
                handleKeyDown
              }
              disabled={loading}
            />

            <button
              className="ask-button"
              onClick={askQuestion}
              disabled={
                loading ||
                !question.trim()
              }
            >
              {loading
                ? "..."
                : "Ask"}
            </button>

          </div>

        </main>

      </div>

    </div>
  )
}

export default App