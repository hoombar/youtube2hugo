# Frame Selection Algorithm Comparison

## Old Temporal Frame Selection Algorithm

```mermaid
flowchart TD
    A[Video Input] --> B[Extract Frames at Regular Intervals]
    B --> C[Every 15s + Dense Sampling]
    C --> D[Group Transcript into Time-Based Paragraphs]
    D --> E[60-120 second paragraphs based on duration]
    
    F[For Each Paragraph] --> G[Find Frames by Timestamp Overlap]
    G --> H{Frame timestamp within<br/>paragraph time ± 5s?}
    H -->|Yes| I[Add to Candidate Frames]
    H -->|No| F
    I --> J[Score Frames by Quality]
    J --> K[Face ratio, blur detection, etc.]
    K --> L[Select Best Frames for Paragraph]
    L --> M[Remove Similar Images]
    M --> N[Place Images After Paragraph Text]
    N --> O{More Paragraphs?}
    O -->|Yes| F
    O -->|No| P[Complete Blog Post]
    
    style B fill:#ffcccc
    style G fill:#ffcccc
    style H fill:#ffcccc
    style L fill:#ffcccc
```

### Characteristics:
- **Time-driven**: Frames selected purely by timestamp overlap
- **Paragraph boundaries**: Fixed 60-120 second chunks regardless of content
- **Content-blind**: No analysis of what's being discussed
- **Predictable placement**: Even distribution across timeline
- **Fast processing**: No AI analysis required

---

## New Semantic Frame Selection Algorithm

```mermaid
flowchart TD
    A[Video Input] --> B[Extract Candidate Frames]
    B --> C[Analyze Transcript with Gemini AI]
    C --> D[Identify Semantic Sections]
    D --> E[Technology demos, concepts, workflows, etc.]
    
    F[For Each Frame] --> G[Analyze Frame Content with Gemini]
    G --> H[Determine Visual Content Type]
    H --> I[Code, UI, diagram, talking head, etc.]
    I --> J[Score Frame Relevance to Sections]
    J --> K[Content matching, visual diversity, etc.]
    
    L[Group High-Scoring Frames by Section] --> M[Create Section-Frame Mappings]
    M --> N[Generate Content with Semantic Context]
    
    N --> O[For Each Paragraph]
    O --> P[Find Semantically Relevant Frames]
    P --> Q{Frames match paragraph<br/>topic + timeframe?}
    Q -->|Yes| R[Add Frames with Section Context]
    Q -->|No| O
    R --> S[Generate Enhanced Alt Text]
    S --> T[Include Section Title Information]
    T --> U{More Paragraphs?}
    U -->|Yes| O
    U -->|No| V[Add Orphaned Frames]
    V --> W[Complete Blog Post with Rich Context]
    
    style C fill:#ccffcc
    style G fill:#ccffcc
    style J fill:#ccffcc
    style P fill:#ccffcc
    style S fill:#ccffcc
```

### Characteristics:
- **Content-aware**: AI analyzes both transcript and visual content
- **Intelligent matching**: Frames selected based on topic relevance
- **Flexible sections**: Content grouped by semantic meaning
- **Rich context**: Section titles and detailed descriptions
- **Enhanced quality**: Better content-to-visual alignment

---

## Key Algorithmic Differences

| Aspect | Temporal Algorithm | Semantic Algorithm |
|--------|-------------------|-------------------|
| **Frame Selection** | Time-based overlap | Content relevance scoring |
| **Paragraph Grouping** | Fixed duration (60-120s) | Topic-based sections |
| **Content Analysis** | None | Gemini AI analysis |
| **Frame Scoring** | Visual quality only | Content relevance + quality |
| **Placement Logic** | Timestamp proximity | Semantic matching |
| **Context Information** | Minimal alt text | Rich section context |
| **Processing Speed** | Fast | Slower (AI analysis) |
| **Content Quality** | Hit-or-miss | Highly relevant |

---

## Frame Placement Comparison

### Temporal Approach:
```
Paragraph 1 (0-60s): "Welcome to this tutorial..."
→ Frame at 45s (whatever was on screen)

Paragraph 2 (60-120s): "First, let's configure the settings..."
→ Frame at 95s (might show unrelated content)

Paragraph 3 (120-180s): "Now we'll implement the function..."
→ Frame at 150s (could be showing previous step)
```

### Semantic Approach:
```
Paragraph 1: "Welcome to this tutorial..."
→ No frames (introduction, no visual content needed)

Paragraph 2: "First, let's configure the settings..."
→ Frame from "Configuration" section (shows actual settings screen)

Paragraph 3: "Now we'll implement the function..."
→ Frame from "Code Implementation" section (shows relevant code)
```

The semantic approach ensures that frames actually illustrate the concepts being discussed, rather than just happening to occur at the same time.