package com.careermatcher.backend;

import java.util.Map;

import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;

/** Forwards conversational career-guidance requests to the FastAPI ML service. */
@RestController
@RequestMapping("/api/career-guidance")
@CrossOrigin(origins = "http://localhost:4200")
public class CareerGuidanceController {

    private final RestClient mlClient;

    public CareerGuidanceController(RestClient.Builder builder) {
        this.mlClient = builder.clone()
                .baseUrl("http://localhost:8000")
                .build();
    }

    @PostMapping(
            value = "/chat",
            consumes = MediaType.APPLICATION_JSON_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE)
    public Map<String, Object> chat(@RequestBody Map<String, Object> payload) {
        Map<String, Object> response = mlClient.post()
                .uri("/api/career-guidance/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .body(payload)
                .retrieve()
                .body(new ParameterizedTypeReference<Map<String, Object>>() {});

        if (response == null) {
            throw new IllegalStateException("ML service returned an empty chat response.");
        }
        return response;
    }
}
