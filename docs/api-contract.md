# Career Matcher API Contract

## ML Service

### POST /parse
Request: `{ "text": "..." }`

Response:
```json
{
  "skills": [],
  "education": [],
  "experience": []
}
```

### POST /match
Request:
```json
{
  "skills": ["Python", "SQL"],
  "experience": [],
  "top_k": 5
}
```

Response:
```json
{
  "occupations": [
    {
      "code": "15-2051.00",
      "title": "Data Scientists",
      "score": 0.87,
      "missing_skills": [
        {"skill": "Statistical Analysis", "importance": 82}
      ]
    }
  ]
}
```

### POST /guidance
Request:
```json
{
  "occupation": {"code": "15-2051.00", "title": "Data Scientists"},
  "missing_skills": ["Statistical Analysis"]
}
```

Response:
```json
{
  "occupation": "Data Scientists",
  "roadmap": []
}
```

## Architecture

React -> Spring Boot -> FastAPI ML Service

The frontend should not call the ML service directly.
