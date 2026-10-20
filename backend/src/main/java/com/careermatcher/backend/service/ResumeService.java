package com.careermatcher.backend.service;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

@Service
public class ResumeService {

    private final RestClient restClient;

    public ResumeService(RestClient.Builder restClientBuilder) {

        this.restClient = restClientBuilder
                .baseUrl("http://localhost:8000")
                .build();
    }

    public String processResume(MultipartFile resume) {

        if (resume == null || resume.isEmpty()) {
            return "No resume uploaded";
        }

        try {
            System.out.println("================================");
            System.out.println("Sending file to ML service");
            System.out.println("Filename: " + resume.getOriginalFilename());
            System.out.println("Size: " + resume.getSize() + " bytes");
            System.out.println("================================");

            // Safely extract bytes so the stream isn't closed prematurely
            byte[] fileBytes = resume.getBytes();

            // Wrap in an anonymous ByteArrayResource to ensure filename is passed correctly
            ByteArrayResource resource = new ByteArrayResource(fileBytes) {
                @Override
                public String getFilename() {
                    return resume.getOriginalFilename();
                }
            };

            MultipartBodyBuilder builder = new MultipartBodyBuilder();

            builder.part("resume", resource)
                    .contentType(
                            resume.getContentType() != null
                                    ? MediaType.parseMediaType(resume.getContentType())
                                    : MediaType.APPLICATION_OCTET_STREAM
                    );

            // NOTE: do NOT set .contentType(MediaType.MULTIPART_FORM_DATA) here.
            // RestClient derives the correct multipart/form-data;boundary=... header
            // automatically from builder.build(). Setting it explicitly strips the
            // boundary and breaks parsing on the server side.
            String response = restClient.post()
                    .uri("/api/resume/analyze")
                    .body(builder.build())
                    .retrieve()
                    .body(String.class);

            System.out.println("================================");
            System.out.println("ML SERVICE RESPONSE");
            System.out.println(response);
            System.out.println("================================");

            return response;

        } catch (HttpClientErrorException e) {
            System.err.println("FastAPI rejected the request: " + e.getStatusCode());
            System.err.println("FastAPI Error Body: " + e.getResponseBodyAsString());

            return "ML Service rejected request: " + e.getResponseBodyAsString();

        } catch (Exception e) {
            e.printStackTrace();
            return "Error while processing resume: " + e.getMessage();
        }
    }
}