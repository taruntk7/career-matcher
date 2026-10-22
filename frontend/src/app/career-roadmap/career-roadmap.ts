import { Component, Input } from '@angular/core';

export interface CareerRoadmapStrength {
  title: string;
  evidence: string;
  relevance: string;
}

export interface CareerRoadmapPriority {
  title: string;
  reason: string;
  resume_evidence: string;
  actions: string[];
  timeframe: string;
  roadmap_topics: string[];
}

export interface CareerRoadmapWeek {
  week: number;
  focus: string;
  actions: string[];
}

export interface CareerRoadmap {
  occupation: string;
  summary: string;
  strengths: CareerRoadmapStrength[];
  priorities: CareerRoadmapPriority[];
  four_week_plan: CareerRoadmapWeek[];
  next_step: string;
  sources: {
    onet: { title: string | null; code: string | null } | null;
    roadmap: {
      name: string;
      url: string;
      topics: { name: string; subtopics: string[] }[];
    } | null;
  };
}

@Component({
  selector: 'app-career-roadmap',
  standalone: true,
  templateUrl: './career-roadmap.html',
  styleUrl: './career-roadmap.css',
})
export class CareerRoadmapComponent {
  @Input() roadmap: CareerRoadmap | null = null;
  @Input() status = 'unavailable';
}
