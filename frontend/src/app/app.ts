import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RouterOutlet } from '@angular/router';

interface CareerRecommendation {
  rank: number;
  code: string;
  title: string;
  description: string;
  final_score: number;
  semantic_score: number;
  capability_score: number;
  essential_score: number;
  transferable_score: number;
  technical_score: number;

  reasons: string[];

  top_technical_strengths: {
    name: string;
    onet_name: string;
    specificity: number;
  }[];

  top_capability_strengths: {
    name: string;
    importance: number;
    evidence_strength: number;
    evidence: string[];
  }[];

  potential_capability_gaps: {
    name: string;
    type: string;
    importance: number;
    priority: number;
    evidence_status: string;
  }[];
}

interface CareerApiResponse {
  filename: string;
  recommendations: CareerRecommendation[];
  career_guidance: string | null;
  career_guidance_status: string;
  career_guidance_error?: string | null;
  message?: string;
}

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html'
})
export class App implements OnInit {


  recommendations: CareerRecommendation[] = [];

careerGuidance: string | null = null;
careerGuidanceStatus = 'unavailable';

isUploading = false;
uploadError = '';
  // =========================
  // Typing Animation
  // =========================

  typedHeading = '';

  private heading = 'Find Where Your Skills Can Take You.';
  private charIndex = 0;
  private deleting = false;
  private typingTimer: ReturnType<typeof setInterval> | undefined;

  // =========================
  // Resume Upload
  // =========================

  selectedFile: File | null = null;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef
  ) {}

  // =========================
  // Page Initialization
  // =========================

  ngOnInit(): void {
    this.startTyping();
  }

  // =========================
  // Typing Animation
  // =========================

  private startTyping(): void {

    this.typingTimer = setInterval(() => {

      // -------------------------
      // Typing
      // -------------------------

      if (!this.deleting) {

        this.typedHeading = this.heading.substring(
          0,
          this.charIndex + 1
        );

        this.charIndex++;

        // Tell Angular to update the screen
        this.cdr.detectChanges();

        // Finished typing
        if (this.charIndex === this.heading.length) {

          clearInterval(this.typingTimer);

          setTimeout(() => {
            this.deleting = true;
            this.startTyping();
          }, 2500);
        }

      }

      // -------------------------
      // Deleting
      // -------------------------

      else {

        this.typedHeading = this.heading.substring(
          0,
          this.charIndex - 1
        );

        this.charIndex--;

        // Tell Angular to update the screen
        this.cdr.detectChanges();

        // Finished deleting
        if (this.charIndex === 0) {

          clearInterval(this.typingTimer);

          this.deleting = false;

          setTimeout(() => {
            this.startTyping();
          }, 500);
        }
      }

    }, this.deleting ? 40 : 70);
  }

  // =========================
  // Resume Selection
  // =========================

  onFileSelected(event: Event): void {

    const input = event.target as HTMLInputElement;

    if (input.files && input.files.length > 0) {

      this.selectedFile = input.files[0];

      console.log(
        'Selected file:',
        this.selectedFile.name
      );

      this.uploadResume();
    }
  }

  // =========================
  // Resume Upload
  // =========================

  uploadResume(): void {
  if (!this.selectedFile) {
    this.uploadError = 'Please select a resume.';
    return;
  }

  const formData = new FormData();
  formData.append('resume', this.selectedFile);

  this.isUploading = true;
  this.uploadError = '';
  this.recommendations = [];
  this.careerGuidance = null;

  this.http.post(
    'http://localhost:8080/api/resume/upload',
    formData,
    { responseType: 'text' }
  ).subscribe({
    next: (responseText: string) => {
      try {
        // Convert the JSON string into a JavaScript object.
        const response: CareerApiResponse =
          JSON.parse(responseText);

        // Dynamically populate the three career cards.
        this.recommendations =
          (response.recommendations ?? []).slice(0, 3);

        this.careerGuidance =
          response.career_guidance ?? null;

        this.careerGuidanceStatus =
          response.career_guidance_status ?? 'unavailable';

        console.log('Career recommendations:', this.recommendations);

      } catch (error) {
        console.error('Could not parse API JSON:', error);
        this.uploadError =
          'The server response could not be read as JSON.';
      } finally {
        this.isUploading = false;
      }
    },

    error: (error) => {
      console.error('Resume upload failed:', error);
      this.uploadError =
        'Unable to analyze your resume. Please try again.';
      this.isUploading = false;
    }
  });
}
}