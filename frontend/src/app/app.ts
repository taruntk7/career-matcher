import { Component, OnInit, ChangeDetectorRef, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RouterOutlet } from '@angular/router';
import { CareerRoadmapComponent, type CareerRoadmap } from './career-roadmap/career-roadmap';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

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

  technical_mismatches?: (string | { name?: string })[];

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
  explicit_skills?: string[];
  career_roadmap?: CareerRoadmap | null;
  career_roadmap_status?: string;
  // Backward-compatible summary fields from the ML service.
  career_guidance: string | null;
  career_guidance_status: string;
  career_guidance_error?: string | null;
  message?: string;
}

type ChatRole = 'user' | 'assistant';

interface CareerChatMessage {
  role: ChatRole;
  content: string;
  /** Sanitized rendered HTML, calculated once when an assistant reply arrives. */
  html?: string;
  isError?: boolean;
}

interface CareerChatResponse {
  answer: string;
  occupation: string;
  source: { title: string | null; code: string | null };
  retrieved_sources: { title: string | null; code: string | null }[];
}

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, CareerRoadmapComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {


  recommendations: CareerRecommendation[] = [];
  resumeExplicitSkills: string[] = [];

careerGuidance: string | null = null;
careerGuidanceHtml = '';
careerGuidanceStatus = 'unavailable';
careerRoadmap: CareerRoadmap | null = null;
careerRoadmapStatus = 'unavailable';

  // Conversational RAG chat. The first recommendation supplies the context.
  chatOpen = false;
  chatDraft = '';
  chatSending = false;
  chatError = '';
  chatMessages: CareerChatMessage[] = [
    {
      role: 'assistant',
      content: 'Hi! I’m your career assistant. Ask me about your top career match, potential skill gaps, or what to learn next.'
    }
  ];

  @ViewChild('chatMessagesContainer')
  private chatMessagesContainer?: ElementRef<HTMLDivElement>;

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


  toggleChat(): void {
    if (this.recommendations.length === 0) return;
    this.chatOpen = !this.chatOpen;
    if (this.chatOpen) this.scrollChatToBottom();
  }


  /** Convert model-generated Markdown to sanitized HTML for display. */
  renderMarkdown(markdown: string | null | undefined): string {
    if (!markdown) return '';

    const html = marked.parse(markdown, {
      gfm: true,
      breaks: false,
      async: false
    }) as string;

    return DOMPurify.sanitize(html, {
      USE_PROFILES: { html: true }
    });
  }

  onChatInput(event: Event): void {
    this.chatDraft = (event.target as HTMLTextAreaElement).value;
  }

  onChatKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendChatMessage();
    }
  }

  onChatSubmit(event: Event): void {
    event.preventDefault();
    this.sendChatMessage();
  }

  askSuggestedQuestion(question: string): void {
    this.chatDraft = question;
    this.sendChatMessage();
  }

  sendChatMessage(): void {
    const question = this.chatDraft.trim();
    const recommendation = this.recommendations[0];

    if (!question || !recommendation || this.chatSending) return;

    // Send only actual prior turns, not the initial greeting or error bubbles.
    const history = this.chatMessages
      .slice(1)
      .filter(message => !message.isError)
      .map(message => ({ role: message.role, content: message.content }));

    const payload = {
      question,
      occupation: recommendation.title,
      occupation_code: recommendation.code,
      capability_gaps: recommendation.potential_capability_gaps ?? [],
      technical_mismatches: recommendation.technical_mismatches ?? [],
      capability_strengths: recommendation.top_capability_strengths ?? [],
      technical_strengths: recommendation.top_technical_strengths ?? [],
      resume_explicit_skills: this.resumeExplicitSkills,
      history
    };

    this.chatMessages = [
      ...this.chatMessages,
      { role: 'user', content: question }
    ];
    this.chatDraft = '';
    this.chatError = '';
    this.chatSending = true;
    this.scrollChatToBottom();

    this.http.post<CareerChatResponse>(
      'http://localhost:8080/api/career-guidance/chat',
      payload
    ).subscribe({
      next: (response) => {
        const answer = response?.answer || 'I could not generate an answer just now.';
        this.chatMessages = [
          ...this.chatMessages,
          {
            role: 'assistant',
            content: answer,
            html: this.renderMarkdown(answer)
          }
        ];
        this.chatSending = false;
        this.scrollChatToBottom();
      },
      error: (error) => {
        console.error('Career chat request failed:', error);
        const detail = error?.error?.detail;
        this.chatError = typeof detail === 'string'
          ? detail
          : 'I couldn’t reach the career assistant. Please try again.';
        this.chatSending = false;
        this.scrollChatToBottom();
      }
    });
  }

  private scrollChatToBottom(): void {
    setTimeout(() => {
      const container = this.chatMessagesContainer?.nativeElement;
      if (container) container.scrollTop = container.scrollHeight;
    }, 0);
  }

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
  this.resumeExplicitSkills = [];
  this.careerGuidance = null;
  this.careerGuidanceHtml = '';
  this.careerRoadmap = null;
  this.careerRoadmapStatus = 'unavailable';

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
        this.resumeExplicitSkills = response.explicit_skills ?? [];

        this.careerRoadmap = response.career_roadmap ?? null;
        this.careerRoadmapStatus =
          response.career_roadmap_status ?? response.career_guidance_status ?? 'unavailable';

        // Keep these legacy fields for compatibility with the current API.
        this.careerGuidance = response.career_guidance ?? null;
        this.careerGuidanceHtml = this.renderMarkdown(this.careerGuidance);
        this.careerGuidanceStatus = response.career_guidance_status ?? 'unavailable';

        // Start a clean chat when a new resume/profile is analyzed.
        this.chatMessages = [{
          role: 'assistant',
          content: 'Hi! I’m your career assistant. Ask me about your top career match, potential skill gaps, or what to learn next.'
        }];
        this.chatDraft = '';
        this.chatError = '';
        this.chatOpen = false;

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
