# Text Summarization System Flowchart

## System Architecture Flow

```mermaid
graph TD
    A[User Input Text] --> B{Text Validation}
    B -->|Valid| C[Preprocess Text]
    B -->|Invalid| D[Return Error Message]
    
    C --> E[Check Model Availability]
    E -->|Available| F[Load BART Model]
    E -->|Not Available| G[Use Fallback Method]
    
    F --> H[Tokenize Input]
    H --> I[Generate Summary with BART]
    I --> J[Post-process Summary]
    
    G --> K[Extract Sentences]
    K --> L[Select First 3 Sentences]
    L --> M[Truncate to Max Length]
    
    J --> N[Return Summary]
    M --> N
    
    N --> O[Display Result to User]
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style F fill:#fff3e0
    style G fill:#fce4ec
```

## API Service Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit Frontend
    participant A as FastAPI Backend
    participant M as Summarizer Module
    participant B as BART Model
    
    U->>S: Enter text & parameters
    S->>S: Validate input
    S->>A: POST /summarize
    A->>A: Parse request
    A->>M: Call summarize_text()
    M->>M: Check model availability
    
    alt Model Available
        M->>B: Load BART pipeline
        B->>M: Model ready
        M->>B: Generate summary
        B->>M: Return summary
    else Model Not Available
        M->>M: Use fallback method
    end
    
    M->>A: Return summary
    A->>S: JSON response
    S->>U: Display summary
```

## Component Interaction Flow

```mermaid
graph LR
    subgraph "Frontend Layer"
        UI[Streamlit UI]
        INPUT[Text Input Area]
        PARAMS[Length Parameters]
        BUTTON[Summarize Button]
    end
    
    subgraph "API Layer"
        API[FastAPI Server]
        CORS[CORS Middleware]
        ENDPOINT[/summarize]
    end
    
    subgraph "Core Engine"
        ENGINE[Summarizer Module]
        MODEL[BART Model]
        FALLBACK[Fallback Method]
    end
    
    UI --> INPUT
    UI --> PARAMS
    UI --> BUTTON
    BUTTON --> API
    API --> CORS
    CORS --> ENDPOINT
    ENDPOINT --> ENGINE
    ENGINE --> MODEL
    ENGINE --> FALLBACK
    MODEL --> ENGINE
    FALLBACK --> ENGINE
    ENGINE --> ENDPOINT
    ENDPOINT --> UI
    
    style UI fill:#e3f2fd
    style API fill:#f3e5f5
    style ENGINE fill:#e8f5e8
```

## Data Processing Flow

```mermaid
flowchart TD
    START[Raw Text Input] --> VALIDATE{Text Validation}
    VALIDATE -->|Empty/Invalid| ERROR[Return Error]
    VALIDATE -->|Valid| CLEAN[Clean & Normalize Text]
    
    CLEAN --> CHECK{Model Available?}
    CHECK -->|Yes| LOAD[Load BART Pipeline]
    CHECK -->|No| FALLBACK[Use Sentence-based Fallback]
    
    LOAD --> TOKENIZE[Tokenize Input Text]
    TOKENIZE --> GENERATE[Generate Summary]
    GENERATE --> POST[Post-process Summary]
    
    FALLBACK --> SPLIT[Split into Sentences]
    SPLIT --> SELECT[Select First 3 Sentences]
    SELECT --> TRUNCATE[Truncate to Max Length]
    
    POST --> OUTPUT[Final Summary]
    TRUNCATE --> OUTPUT
    
    OUTPUT --> DISPLAY[Display to User]
    
    style START fill:#ffebee
    style OUTPUT fill:#e8f5e8
    style ERROR fill:#ffcdd2
    style LOAD fill:#fff3e0
    style FALLBACK fill:#fce4ec
```

## Error Handling Flow

```mermaid
graph TD
    INPUT[User Input] --> CHECK1{Text Empty?}
    CHECK1 -->|Yes| ERR1[Show Warning: Please provide text]
    CHECK1 -->|No| CHECK2{Text Too Long?}
    
    CHECK2 -->|Yes| TRUNCATE[Truncate Text]
    CHECK2 -->|No| CHECK3{Model Available?}
    
    TRUNCATE --> CHECK3
    CHECK3 -->|No| FALLBACK[Use Fallback Method]
    CHECK3 -->|Yes| CHECK4{API Call Success?}
    
    FALLBACK --> OUTPUT1[Return Fallback Summary]
    CHECK4 -->|No| ERR2[Show Error: API call failed]
    CHECK4 -->|Yes| OUTPUT2[Return BART Summary]
    
    ERR1 --> END[End]
    ERR2 --> END
    OUTPUT1 --> END
    OUTPUT2 --> END
    
    style ERR1 fill:#ffcdd2
    style ERR2 fill:#ffcdd2
    style OUTPUT1 fill:#c8e6c9
    style OUTPUT2 fill:#c8e6c9
```

## Deployment Architecture Flow

```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Local Development]
        TEST[Testing Phase]
    end
    
    subgraph "Production Environment"
        PROD[Production Server]
        GPU[GPU Acceleration]
        CPU[CPU Fallback]
    end
    
    subgraph "User Access"
        WEB[Web Browser]
        API_CLIENT[API Client]
    end
    
    DEV --> TEST
    TEST --> PROD
    PROD --> GPU
    PROD --> CPU
    GPU --> WEB
    CPU --> WEB
    GPU --> API_CLIENT
    CPU --> API_CLIENT
    
    style DEV fill:#e1f5fe
    style PROD fill:#e8f5e8
    style GPU fill:#fff3e0
    style CPU fill:#fce4ec
```

## Performance Optimization Flow

```mermaid
graph LR
    REQUEST[Incoming Request] --> CACHE{Cache Available?}
    CACHE -->|Yes| CACHED[Return Cached Result]
    CACHE -->|No| GPU_CHECK{GPU Available?}
    
    GPU_CHECK -->|Yes| GPU_PROCESS[Process on GPU]
    GPU_CHECK -->|No| CPU_PROCESS[Process on CPU]
    
    GPU_PROCESS --> RESULT[Generate Summary]
    CPU_PROCESS --> RESULT
    
    RESULT --> STORE[Store in Cache]
    STORE --> RETURN[Return Result]
    CACHED --> RETURN
    
    style GPU_PROCESS fill:#c8e6c9
    style CPU_PROCESS fill:#ffecb3
    style CACHED fill:#e1f5fe
```

## Security and Privacy Flow

```mermaid
graph TD
    INPUT[User Input] --> SANITIZE[Sanitize Input]
    SANITIZE --> FILTER[Content Filtering]
    FILTER --> PII_CHECK{PII Detection}
    
    PII_CHECK -->|Detected| WARN[Show Warning]
    PII_CHECK -->|Clean| PROCESS[Process Text]
    
    PROCESS --> MEMORY[In-Memory Processing]
    MEMORY --> NO_STORE[No Persistent Storage]
    NO_STORE --> RETURN[Return Summary]
    
    WARN --> USER_CHOICE{User Continues?}
    USER_CHOICE -->|Yes| PROCESS
    USER_CHOICE -->|No| CANCEL[Cancel Operation]
    
    style SANITIZE fill:#e8f5e8
    style WARN fill:#fff3e0
    style NO_STORE fill:#e1f5fe
    style CANCEL fill:#ffcdd2
```

## Monitoring and Logging Flow

```mermaid
graph TD
    REQUEST[API Request] --> LOG[Log Request]
    LOG --> METRICS[Collect Metrics]
    METRICS --> PROCESS[Process Request]
    
    PROCESS --> SUCCESS{Success?}
    SUCCESS -->|Yes| LOG_SUCCESS[Log Success]
    SUCCESS -->|No| LOG_ERROR[Log Error]
    
    LOG_SUCCESS --> METRICS_SUCCESS[Update Success Metrics]
    LOG_ERROR --> METRICS_ERROR[Update Error Metrics]
    
    METRICS_SUCCESS --> MONITOR[Monitoring Dashboard]
    METRICS_ERROR --> MONITOR
    
    MONITOR --> ALERT{Threshold Exceeded?}
    ALERT -->|Yes| NOTIFY[Send Alert]
    ALERT -->|No| CONTINUE[Continue Monitoring]
    
    style LOG fill:#e1f5fe
    style MONITOR fill:#e8f5e8
    style NOTIFY fill:#ffcdd2
```

## Complete System Overview

```mermaid
graph TB
    subgraph "User Interface"
        UI[Streamlit Web App]
        INPUT[Text Input]
        CONFIG[Configuration]
    end
    
    subgraph "API Gateway"
        FASTAPI[FastAPI Server]
        CORS[CORS Middleware]
        VALIDATE[Request Validation]
    end
    
    subgraph "Core Processing"
        SUMMARIZER[Summarizer Module]
        BART[BART Model]
        FALLBACK[Fallback Method]
    end
    
    subgraph "Infrastructure"
        GPU[GPU Acceleration]
        CPU[CPU Processing]
        CACHE[Response Cache]
    end
    
    subgraph "Monitoring"
        LOGS[Application Logs]
        METRICS[Performance Metrics]
        ALERTS[Error Alerts]
    end
    
    UI --> INPUT
    INPUT --> CONFIG
    CONFIG --> FASTAPI
    FASTAPI --> CORS
    CORS --> VALIDATE
    VALIDATE --> SUMMARIZER
    SUMMARIZER --> BART
    SUMMARIZER --> FALLBACK
    BART --> GPU
    BART --> CPU
    FALLBACK --> CPU
    GPU --> CACHE
    CPU --> CACHE
    CACHE --> FASTAPI
    FASTAPI --> UI
    
    SUMMARIZER --> LOGS
    FASTAPI --> METRICS
    LOGS --> ALERTS
    METRICS --> ALERTS
    
    style UI fill:#e3f2fd
    style FASTAPI fill:#f3e5f5
    style SUMMARIZER fill:#e8f5e8
    style GPU fill:#fff3e0
    style CPU fill:#fce4ec
    style LOGS fill:#f1f8e9
```

## Usage Instructions

To view these flowcharts:

1. **Online**: Copy the Mermaid code and paste it into [Mermaid Live Editor](https://mermaid.live/)
2. **VS Code**: Install the Mermaid Preview extension
3. **GitHub**: These will render automatically in GitHub markdown files
4. **Documentation**: Use with MkDocs, GitBook, or other documentation platforms

## Flowchart Descriptions

- **System Architecture Flow**: Shows the main processing pipeline from input to output
- **API Service Flow**: Sequence diagram showing the interaction between components
- **Component Interaction Flow**: How different system components communicate
- **Data Processing Flow**: Detailed data transformation steps
- **Error Handling Flow**: How the system handles various error conditions
- **Deployment Architecture Flow**: System deployment and environment setup
- **Performance Optimization Flow**: Caching and GPU acceleration strategies
- **Security and Privacy Flow**: Data protection and privacy measures
- **Monitoring and Logging Flow**: System monitoring and alerting
- **Complete System Overview**: High-level view of the entire system architecture

