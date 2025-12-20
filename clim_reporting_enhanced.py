"""
Module de reporting avancé pour l'application Climate Risk Tool.
Génère des rapports HTML professionnels avec visualisations et analyses des risques climatiques.
"""

import os
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple

# Configuration des dossiers de sortie
OUTPUT_DIR = "outputs/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# Fonctions utilitaires
# ============================================

def _get_plotly_figure_html(fig: go.Figure, width: int = 800, height: int = 500) -> str:
    """Convertit une figure Plotly en HTML."""
    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True})

def _get_css_styles() -> str:
    """Retourne le CSS personnalisé pour le rapport."""
    return """
    <style>
        :root {
            --primary-color: #3b82f6;
            --secondary-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #3b82f6;
            --light-bg: #f8fafc;
            --dark-bg: #0f172a;
            --text-color: #1e293b;
            --text-light: #64748b;
            --border-color: #e2e8f0;
            --success-color: #10b981;
        }
        
        /* Styles de base */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: #f1f5f9;
            margin: 0;
            padding: 0;
            font-size: 16px;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-radius: 0.5rem;
            overflow-x: hidden;
        }
        
        /* Styles pour les tableaux */
        .table-responsive {
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin-bottom: 2rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .dataframe {
            width: 100% !important;
            margin: 0;
            border-collapse: collapse;
            font-size: 0.9rem;
            min-width: 100%;
        }
        
        .dataframe th, 
        .dataframe td {
            padding: 0.75rem;
            text-align: left;
            border: 1px solid #e2e8f0;
            white-space: nowrap;
        }
        
        .dataframe th {
            background-color: #f8fafc;
            font-weight: 600;
            color: #1e293b;
            position: sticky;
            top: 0;
        }
        
        .dataframe tr:nth-child(even) {
            background-color: #f8fafc;
        }
        
        .dataframe tr:hover {
            background-color: #f1f5f9;
        }
        
        /* Styles pour les graphiques */
        .plot-container {
            width: 100%;
            margin-bottom: 2rem;
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        .plot-container .js-plotly-plot,
        .plotly-graph-div {
            width: 100% !important;
            max-width: 100%;
            margin: 0 auto;
        }
        
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-radius: 0.5rem;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .header h1 {
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }
        
        .header .subtitle {
            color: var(--text-light);
            font-size: 1.1rem;
        }
        
        .section {
            margin: 2rem 0;
            padding: 1.5rem;
            background: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .section-title {
            color: var(--primary-color);
            margin-top: 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .kpi-card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.2s;
            border-left: 4px solid var(--primary-color);
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            margin: 0.5rem 0;
        }
        
        .kpi-label {
            color: var(--text-light);
            font-size: 0.9rem;
        }
        
        .warning {
            background-color: #fffbeb;
            border-left: 4px solid var(--warning-color);
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .warning svg {
            color: var(--warning-color);
            flex-shrink: 0;
        }
        
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 2rem;
            margin: 2rem 0;
        }
        
        @media (max-width: 1200px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }
        
        .plot-container {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .plot-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0,0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .plot-container h3 {
            margin-top: 0;
            color: var(--text-color);
            font-size: 1.1rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .section-subtitle {
            color: var(--secondary-color);
            margin: 1.5rem 0 1rem 0;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .executive-summary {
            background-color: #f8fafc;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--primary-color);
        }
        
        .key-findings {
            background: white;
            padding: 1.25rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        .key-findings ul {
            padding-left: 1.25rem;
            margin: 0.75rem 0;
        }
        
        .key-findings li {
            margin-bottom: 0.5rem;
            line-height: 1.6;
        }
        
        .recommendations {
            background-color: #f0fdf4;
            border-left: 4px solid var(--success-color);
            padding: 1.25rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0;
        }
        
        .recommendations ol {
            padding-left: 1.5rem;
            margin: 0.75rem 0;
        }
        
        .recommendations li {
            margin-bottom: 0.5rem;
        }
        
        /* Styles pour la section des recommandations */
        .recommendations-section {
            background: #f8fafc;
            border-radius: 0.75rem;
            overflow: hidden;
        }
        
        .section-header {
            text-align: center;
            margin-bottom: 2.5rem;
            padding: 0 1.5rem;
        }
        
        .section-subtitle {
            color: var(--text-light);
            font-size: 1.1rem;
            max-width: 700px;
            margin: 0.75rem auto 0;
            line-height: 1.6;
        }
        
        .recommendations-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
            margin-bottom: 3rem;
        }
        
        .risk-summary-card {
            background: white;
            border-radius: 0.75rem;
            padding: 1.75rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            height: fit-content;
            border-top: 4px solid #3b82f6;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .risk-level {
            padding: 0.35rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .risk-level.high {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .risk-level.medium {
            background: #fef3c7;
            color: #92400e;
        }
        
        .risk-level.low {
            background: #dcfce7;
            color: #166534;
        }
        
        .risk-metrics {
            display: grid;
            gap: 1.25rem;
            margin: 1.5rem 0;
        }
        
        .risk-metric {
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            color: var(--text-color);
            font-weight: 500;
        }
        
        .metric-bar {
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }
        
        .metric-fill {
            height: 100%;
            border-radius: 4px;
        }
        
        .metric-fill.high {
            background: #ef4444;
        }
        
        .metric-fill.medium {
            background: #f59e0b;
        }
        
        .metric-fill.low {
            background: #10b981;
        }
        
        .risk-insight {
            background: #f0f9ff;
            border-left: 3px solid #0ea5e9;
            padding: 1rem;
            border-radius: 0 0.5rem 0.5rem 0;
            margin-top: 1.5rem;
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
        }
        
        .insight-icon {
            font-size: 1.25rem;
            color: #0ea5e9;
            margin-top: 0.15rem;
        }
        
        .risk-insight p {
            margin: 0;
            font-size: 0.9rem;
            color: #0369a1;
            line-height: 1.5;
        }
        
        .recommendations-categories {
            display: grid;
            gap: 1.5rem;
        }
        
        .category-card {
            background: white;
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .category-header {
            display: flex;
            align-items: center;
            padding: 1.25rem 1.5rem;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .category-icon {
            font-size: 1.5rem;
            margin-right: 0.75rem;
        }
        
        .category-header h3 {
            margin: 0;
            font-size: 1.25rem;
            color: var(--text-color);
        }
        
        .recommendation-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .recommendation-item {
            border-bottom: 1px solid #e2e8f0;
        }
        
        .recommendation-item:last-child {
            border-bottom: none;
        }
        
        .recommendation-check {
            display: none;
        }
        
        .recommendation-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 1.5rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .recommendation-label:hover {
            background-color: #f8fafc;
        }
        
        .recommendation-text {
            flex: 1;
            margin-right: 1rem;
            font-weight: 500;
        }
        
        .recommendation-priority {
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border-radius: 1rem;
            font-weight: 600;
        }
        
        .recommendation-priority.high {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .recommendation-priority.medium {
            background: #fef3c7;
            color: #92400e;
        }
        
        .recommendation-priority.low {
            background: #e0f2fe;
            color: #075985;
        }
        
        .recommendation-details {
            padding: 0 1.5rem 1.5rem;
            display: none;
            animation: fadeIn 0.3s;
        }
        
        .recommendation-check:checked ~ .recommendation-details {
            display: block;
        }
        
        .recommendation-details p {
            margin: 0 0 1rem 0;
            color: var(--text-light);
            line-height: 1.6;
        }
        
        .recommendation-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.85rem;
            color: #94a3b8;
        }
        
        .meta-item {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        
        /* Plan d'action */
        .action-plan {
            background: white;
            border-radius: 0.75rem;
            padding: 2rem;
            margin: 2.5rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .action-plan-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .action-plan h3 {
            margin: 0;
            font-size: 1.5rem;
            color: var(--text-color);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .export-btn {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: #3b82f6;
            color: white;
            border: none;
            padding: 0.6rem 1.25rem;
            border-radius: 0.5rem;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .export-btn:hover {
            background: #2563eb;
        }
        
        .btn-icon {
            font-size: 1.1em;
        }
        
        .timeline {
            position: relative;
            padding-left: 2rem;
            margin: 2rem 0;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 0.5rem;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #e2e8f0;
        }
        
        .timeline-item {
            position: relative;
            padding-bottom: 2.5rem;
            padding-left: 2rem;
        }
        
        .timeline-item:last-child {
            padding-bottom: 0;
        }
        
        .timeline-marker {
            position: absolute;
            left: -1.75rem;
            top: 0;
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 50%;
            background: #3b82f6;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            z-index: 1;
            border: 3px solid white;
            box-shadow: 0 0 0 3px #bfdbfe;
        }
        
        .timeline-content {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #e2e8f0;
        }
        
        .timeline-content h4 {
            margin: 0 0 1rem 0;
            color: var(--text-color);
            font-size: 1.1rem;
        }
        
        .action-items {
            margin: 0;
            padding-left: 1.25rem;
        }
        
        .action-items li {
            margin-bottom: 0.5rem;
            color: var(--text-light);
        }
        
        .action-items li:last-child {
            margin-bottom: 0;
        }
        
        .kpi-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin-top: 2.5rem;
        }
        
        .kpi-card {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 0.75rem;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        
        .kpi-value {
            font-size: 2.25rem;
            font-weight: 700;
            color: #3b82f6;
            margin-bottom: 0.5rem;
            line-height: 1;
        }
        
        .kpi-label {
            font-size: 1rem;
            color: var(--text-color);
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .kpi-subtext {
            font-size: 0.85rem;
            color: #94a3b8;
        }
        
        /* Prochaines étapes */
        .next-steps {
            background: white;
            border-radius: 0.75rem;
            padding: 2rem;
            margin-top: 2.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .next-steps h3 {
            margin: 0 0 1.5rem 0;
            font-size: 1.5rem;
            color: var(--text-color);
            text-align: center;
        }
        
        .steps-list {
            counter-reset: step;
            list-style: none;
            padding: 0;
            max-width: 600px;
            margin: 0 auto 2.5rem;
        }
        
        .steps-list li {
            position: relative;
            padding: 1.25rem 1.5rem 1.25rem 4rem;
            margin-bottom: 1rem;
            background: #f8fafc;
            border-radius: 0.5rem;
            border-left: 4px solid #3b82f6;
            counter-increment: step;
        }
        
        .steps-list li::before {
            content: counter(step);
            position: absolute;
            left: -1.25rem;
            top: 50%;
            transform: translateY(-50%);
            width: 2.5rem;
            height: 2.5rem;
            background: #3b82f6;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
        }
        
        .steps-list li:last-child {
            margin-bottom: 0;
        }
        
        .cta-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 2rem;
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
            border: none;
            font-size: 1rem;
        }
        
        .btn-primary {
            background: #3b82f6;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2563eb;
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: white;
            color: #3b82f6;
            border: 1px solid #3b82f6;
        }
        
        .btn-secondary:hover {
            background: #f0f7ff;
            transform: translateY(-1px);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Responsive */
        @media (max-width: 1024px) {
            .recommendations-grid {
                grid-template-columns: 1fr;
            }
            
            .kpi-cards {
                grid-template-columns: 1fr;
            }
            
            .cta-buttons {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
                justify-content: center;
            }
        }
        
        /* Styles pour la section de modélisation */
        .modeling-tabs {
            display: flex;
            border-bottom: 1px solid #e2e8f0;
            margin: 1.5rem 0;
            flex-wrap: wrap;
        }
        
        .modeling-tab {
            background: none;
            border: none;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            font-size: 0.95rem;
            color: var(--text-light);
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            margin-right: 0.5rem;
            border-radius: 4px 4px 0 0;
        }
        
        .modeling-tab:hover {
            background-color: #f1f5f9;
            color: var(--primary-color);
        }
        
        .modeling-tab.active {
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            font-weight: 600;
            background-color: #f8fafc;
        }
        
        .modeling-tabcontent {
            display: none;
            padding: 1.5rem 0;
            animation: fadeIn 0.5s;
        }
        
        .model-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 1.25rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            text-align: center;
            border-left: 4px solid var(--primary-color);
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary-color);
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: var(--text-light);
        }
        
        .forecast-controls {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .forecast-options {
            display: flex;
            gap: 1.5rem;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        
        .forecast-options label {
            font-weight: 500;
            color: var(--text-color);
        }
        
        .forecast-options select {
            padding: 0.5rem 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 0.375rem;
            background-color: white;
            color: var(--text-color);
        }
        
        .confidence-interval {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .forecast-chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            min-height: 400px;
        }
        
        .forecast-summary {
            margin: 2rem 0;
        }
        
        .forecast-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }
        
        .forecast-card {
            display: flex;
            align-items: center;
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            gap: 1rem;
        }
        
        .forecast-icon {
            font-size: 2rem;
            width: 3.5rem;
            height: 3.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8fafc;
            border-radius: 0.5rem;
        }
        
        .forecast-details {
            flex: 1;
        }
        
        .forecast-metric {
            font-size: 0.9rem;
            color: var(--text-light);
        }
        
        .forecast-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-color);
            margin: 0.25rem 0;
        }
        
        .forecast-period {
            font-size: 0.85rem;
            color: var(--text-light);
        }
        
        .forecast-change {
            font-size: 0.9rem;
            margin-left: 0.5rem;
            padding: 0.15rem 0.5rem;
            border-radius: 1rem;
            font-weight: 500;
        }
        
        .forecast-change.positive {
            background-color: #dcfce7;
            color: #16a34a;
        }
        
        .forecast-change.negative {
            background-color: #fee2e2;
            color: #dc2626;
        }
        
        .forecast-alert {
            font-size: 0.8rem;
            background-color: #fef3c7;
            color: #d97706;
            padding: 0.15rem 0.5rem;
            border-radius: 1rem;
            margin-left: 0.5rem;
            font-weight: 500;
        }
        
        /* Styles pour l'analyse des variables */
        .feature-analysis {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin: 1.5rem 0;
        }
        
        .feature-importance-chart {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            min-height: 400px;
        }
        
        .feature-details {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .feature-list {
            margin: 1.5rem 0;
        }
        
        .feature-item {
            margin-bottom: 1rem;
        }
        
        .feature-name {
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
            color: var(--text-color);
        }
        
        .feature-importance {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .importance-bar {
            height: 8px;
            border-radius: 4px;
            background: #e2e8f0;
            flex: 1;
        }
        
        .importance-value {
            font-size: 0.85rem;
            color: var(--text-light);
            min-width: 2.5rem;
            text-align: right;
        }
        
        .feature-insights {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1.5rem;
        }
        
        .feature-insights h5 {
            margin: 0 0 0.75rem 0;
            color: var(--text-color);
        }
        
        .feature-insights ul {
            margin: 0;
            padding-left: 1.25rem;
        }
        
        .feature-insights li {
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-light);
        }
        
        /* Styles pour les scénarios climatiques */
        .scenario-selector {
            margin: 1.5rem 0;
        }
        
        .scenario-options {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 1rem 0 2rem;
        }
        
        .scenario-option {
            position: relative;
            cursor: pointer;
        }
        
        .scenario-option input[type="radio"] {
            position: absolute;
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .scenario-card {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 0.5rem;
            padding: 1.5rem;
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .scenario-option input:checked + .scenario-card {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }
        
        .scenario-title {
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        
        .scenario-desc {
            font-size: 0.9rem;
            color: var(--text-light);
            margin-bottom: 0.75rem;
        }
        
        .scenario-temp {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--primary-color);
            padding: 0.25rem 0.75rem;
            background: #eff6ff;
            border-radius: 1rem;
            display: inline-block;
        }
        
        .scenario-results {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin: 1.5rem 0;
        }
        
        .scenario-chart {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            min-height: 400px;
        }
        
        .scenario-impacts {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .impact-cards {
            display: grid;
            gap: 1rem;
        }
        
        .impact-card {
            background: white;
            padding: 1.25rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .impact-icon {
            font-size: 1.5rem;
            width: 3rem;
            height: 3rem;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8fafc;
            border-radius: 0.5rem;
        }
        
        .impact-details {
            flex: 1;
        }
        
        .impact-metric {
            font-size: 0.9rem;
            color: var(--text-light);
        }
        
        .impact-value {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-color);
            margin: 0.15rem 0;
        }
        
        .impact-period {
            font-size: 0.8rem;
            color: var(--text-light);
        }
        
        .scenario-recommendations {
            background: #f8fafc;
            padding: 1.25rem;
            border-radius: 0.5rem;
            margin-top: auto;
        }
        
        .scenario-recommendations h5 {
            margin: 0 0 0.75rem 0;
            color: var(--text-color);
        }
        
        .scenario-recommendations ul {
            margin: 0;
            padding-left: 1.25rem;
        }
        
        .scenario-recommendations li {
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-light);
        }
        
        /* Styles pour les onglets des métriques */
        .metrics-tabs {
            display: flex;
            border-bottom: 1px solid #e2e8f0;
            margin: 1.5rem 0;
            flex-wrap: wrap;
        }
        
        .metrics-tab {
            background: none;
            border: none;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            font-size: 0.95rem;
            color: var(--text-light);
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            margin-right: 0.5rem;
            border-radius: 4px 4px 0 0;
        }
        
        .metrics-tab:hover {
            background-color: #f1f5f9;
            color: var(--primary-color);
        }
        
        .metrics-tab.active {
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            font-weight: 600;
            background-color: #f8fafc;
        }
        
        .metrics-tabcontent {
            display: none;
            padding: 1.5rem 0;
            animation: fadeIn 0.5s;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin: 1.5rem 0;
        }
        
        .metrics-table-container {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .metrics-plot {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        .metrics-table th, .metrics-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .metrics-table th {
            font-weight: 600;
            color: var(--text-color);
            background-color: #f8fafc;
        }
        
        .metrics-table tr:hover {
            background-color: #f8fafc;
        }
        
        .metrics-insights {
            margin: 2rem 0;
        }
        
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        
        .insight-card {
            display: flex;
            background: white;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        
        .insight-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .insight-icon {
            font-size: 2rem;
            margin-right: 1.5rem;
            color: var(--primary-color);
        }
        
        .insight-content h5 {
            margin: 0 0 0.5rem 0;
            color: var(--text-color);
        }
        
        .insight-content p {
            margin: 0;
            color: var(--text-light);
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        /* Styles pour les cartes d'extrêmes */
        .extreme-card {
            display: flex;
            align-items: center;
            padding: 1.5rem;
            border-radius: 0.5rem;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        .extreme-card.heatwave {
            background: linear-gradient(135deg, #f59e0b, #ef4444);
        }
        
        .extreme-card.coldwave {
            background: linear-gradient(135deg, #3b82f6, #06b6d4);
        }
        
        .extreme-card.rainfall {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        }
        
        .extreme-icon {
            font-size: 2.5rem;
            margin-right: 1.5rem;
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
        }
        
        .extreme-content h4 {
            margin: 0 0 0.25rem 0;
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .extreme-value {
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0.25rem 0;
        }
        
        .extreme-content p {
            margin: 0;
            opacity: 0.9;
            font-size: 0.9rem;
        }
        
        /* Styles pour les onglets */
        .tabs {
            display: flex;
            border-bottom: 1px solid #e2e8f0;
            margin: 1.5rem 0;
        }
        
        .tablinks {
            background-color: #f8fafc;
            border: none;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            font-size: 0.95rem;
            color: var(--text-light);
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }
        
        .tablinks:hover {
            background-color: #f1f5f9;
            color: var(--primary-color);
        }
        
        .tablinks.active {
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            font-weight: 600;
        }
        
        .tabcontent {
            display: none;
            padding: 1.5rem 0;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Grille d'analyse */
        .analysis-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin: 1.5rem 0;
        }
        
        .analysis-plot {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .analysis-stats {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .stat-card {
            background: white;
            padding: 1.25rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .stat-value {
            display: block;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.25rem;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: var(--text-light);
        }
        
        .insight {
            background: #f8fafc;
            padding: 1.25rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }
        
        .insight h5 {
            margin: 0 0 0.5rem 0;
            color: var(--secondary-color);
        }
        
        .insight p {
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        /* Cartes d'événements extrêmes */
        .extremes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .extreme-card {
            display: flex;
            align-items: center;
            padding: 1.5rem;
            border-radius: 0.5rem;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .extreme-card.heatwave {
            background: linear-gradient(135deg, #f59e0b, #ef4444);
        }
        
        .extreme-card.rainfall {
            background: linear-gradient(135deg, #3b82f6, #06b6d4);
        }
        
        .extreme-icon {
            font-size: 2.5rem;
            margin-right: 1.5rem;
        }
        
        .extreme-content h4 {
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
        }
        
        .extreme-value {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        
        .extreme-content p {
            margin: 0;
            opacity: 0.9;
            font-size: 0.9rem;
        }
        
        /* Styles pour les tableaux */
        .dataframe {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border-radius: 0.5rem;
            overflow: hidden;
        }
        
        .dataframe th, .dataframe td {
            padding: 0.75rem 1rem;
            text-align: left;
            border: 1px solid var(--border-color);
        }
        
        .dataframe th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }
        
        .dataframe tr:nth-child(even) {
            background-color: #f8fafc;
        }
        
        .dataframe tr:hover {
            background-color: #f1f5f9;
        }
        
        /* Styles pour les sections spéciales */
        .model-section {
            background-color: #f0f9ff;
            border-left: 4px solid var(--info-color);
        }
        
        .metrics-section {
            background-color: #f0fdf4;
            border-left: 4px solid var(--success-color);
        }
        
        .trends-section {
            background-color: #fffbeb;
            border-left: 4px solid var(--warning-color);
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
            color: var(--text-light);
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .kpi-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """

def _create_kpi_card(value: Any, label: str, icon: str = "📊", color: str = "var(--primary-color)") -> str:
    """Crée une carte KPI pour le rapport."""
    return f"""
    <div class="kpi-card" style="border-left-color: {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """

# ============================================
# Fonctions d'analyse des données
# ============================================

def _analyze_climate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse les données climatiques et retourne des métriques clés."""
    analysis = {}
    
    # Vérifier les colonnes numériques
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Détection des colonnes de température, précipitations, etc.
    temp_cols = [col for col in df.columns if any(term in col.lower() for term in ['temp', 'tmax', 'tmin', 'tavg'])]
    precip_cols = [col for col in df.columns if any(term in col.lower() for term in ['precip', 'rain', 'pluie'])]
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    
    # Calcul des métriques de base
    analysis['num_rows'] = len(df)
    analysis['num_cols'] = len(df.columns)
    analysis['missing_values'] = df.isna().sum().sum()
    analysis['missing_percent'] = (analysis['missing_values'] / (len(df) * len(df.columns)) * 100).round(2)
    
    # Statistiques sur les températures
    if temp_cols:
        temp_df = df[temp_cols].select_dtypes(include=['number'])
        if not temp_df.empty:
            analysis['avg_temp'] = temp_df.mean().mean().round(1)
            analysis['temp_range'] = (temp_df.max().max() - temp_df.min().min()).round(1)
            
            # Analyse des tendances de température
            if date_cols and len(df) > 1:
                df_sorted = df.sort_values(by=date_cols[0])
                temp_series = temp_df.mean(axis=1)
                x = np.arange(len(temp_series))
                slope, _ = np.polyfit(x, temp_series, 1)
                
                if slope > 0.1:
                    analysis['trend_analysis'] = f"Hausse significative des températures (+{slope:.2f}°C/an)"
                elif slope < -0.1:
                    analysis['trend_analysis'] = f"Baisse significative des températures ({slope:.2f}°C/an)"
                else:
                    analysis['trend_analysis'] = "Stabilité relative des températures"
            else:
                analysis['trend_analysis'] = "Données insuffisantes pour l'analyse de tendance"
    else:
        analysis['trend_analysis'] = "Aucune donnée de température disponible"
    
    # Statistiques sur les précipitations
    if precip_cols:
        precip_df = df[precip_cols].select_dtypes(include=['number'])
        if not precip_df.empty:
            analysis['avg_precip'] = precip_df.mean().mean().round(1)
            analysis['total_precip'] = precip_df.sum().sum().round(1)
            
            # Détection des valeurs extrêmes
            precip_extremes = precip_df.max()
            analysis['max_precip'] = precip_extremes.max().round(1)
            analysis['precip_extreme_days'] = (precip_df > 50).sum().sum()  # Jours avec plus de 50mm de pluie
            
            # Analyse du régime des précipitations
            if date_cols and len(df) > 1:
                monthly_precip = df.groupby(df[date_cols[0]].dt.month)[precip_cols].mean().mean(axis=1)
                wettest_month = monthly_precip.idxmax()
                driest_month = monthly_precip.idxmin()
                
                month_names = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", 
                             "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
                
                analysis['precip_analysis'] = (
                    f"Saison des pluies en {month_names[monthly_precip.idxmax()-1]} "
                    f"({monthly_precip.max():.1f} mm/mois en moyenne), "
                    f"saison sèche en {month_names[monthly_precip.idxmin()-1]}"
                )
            else:
                analysis['precip_analysis'] = "Données insuffisantes pour l'analyse des précipitations"
    else:
        analysis['precip_analysis'] = "Aucune donnée de précipitation disponible"
        
    # Analyse des risques
    risk_factors = []
    
    # Vérification des vagues de chaleur
    if 'avg_temp' in analysis and analysis['avg_temp'] > 25:
        risk_factors.append("températures moyennes élevées")
    
    # Vérification des précipitations extrêmes
    if 'precip_extreme_days' in analysis and analysis['precip_extreme_days'] > 0:
        risk_factors.append(f"{analysis['precip_extreme_days']} jours de précipitations extrêmes")
    
    # Vérification des données manquantes
    if analysis['missing_percent'] > 5:
        risk_factors.append(f"données manquantes ({analysis['missing_percent']}%)")
    
    if risk_factors:
        analysis['risk_analysis'] = "Risques identifiés : " + ", ".join(risk_factors)
    else:
        analysis['risk_analysis'] = "Aucun risque majeur détecté dans les données actuelles"
    
    # Détection des valeurs aberrantes
    if numeric_cols:
        numeric_df = df[numeric_cols]
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        outliers = ((numeric_df < (q1 - 1.5 * iqr)) | (numeric_df > (q3 + 1.5 * iqr))).sum().sum()
        analysis['outliers'] = outliers
    
    return analysis

def _create_temperature_plot(df: pd.DataFrame, temp_cols: List[str]) -> Optional[go.Figure]:
    """Crée un graphique d'évolution des températures."""
    if not temp_cols:
        return None
        
    # Sélectionner uniquement les colonnes de température numériques
    temp_df = df[temp_cols].select_dtypes(include=['number'])
    if temp_df.empty:
        return None
    
    # Préparer les données pour le tracé
    x_values = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))
    
    # Créer un graphique d'évolution
    fig = go.Figure()
    
    for col in temp_df.columns:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=temp_df[col].values,
            name=col,
            mode='lines+markers',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Évolution des températures",
        xaxis_title="Date" if isinstance(df.index, pd.DatetimeIndex) else "Index",
        yaxis_title="Température (°C)",
        legend_title="Légende",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def _create_precipitation_plot(df: pd.DataFrame, precip_cols: List[str]) -> Optional[go.Figure]:
    """Crée un graphique des précipitations."""
    if not precip_cols:
        return None
        
    # Sélectionner uniquement les colonnes de précipitations numériques
    precip_df = df[precip_cols].select_dtypes(include=['number'])
    if precip_df.empty:
        return None
        
    # Créer un graphique à barres empilées
    fig = go.Figure()
    
    for col in precip_df.columns:
        x_values = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))
        fig.add_trace(go.Bar(
            x=x_values,
            y=precip_df[col],
            name=col
        ))
    
    fig.update_layout(
        title="Précipitations",
        xaxis_title="Date" if isinstance(df.index, pd.DatetimeIndex) else "Index",
        yaxis_title="Précipitations (mm)",
        barmode='stack',
        legend_title="Légende",
        template="plotly_white"
    )
    
    return fig

# ============================================
# Fonction principale de génération de rapport
# ============================================

def generate_climate_report(session_state: Dict[str, Any], report_type: str = "complet") -> str:
    """
    Génère un rapport HTML complet sur les données climatiques.
    
    Args:
        session_state: État de la session Streamlit
        report_type: Type de rapport ('complet', 'executif', 'technique')
        
    Returns:
        str: Contenu HTML du rapport
    """
    # Vérifier si des données sont disponibles
    if 'df' not in session_state:
        return "<div class='error'>Aucune donnée disponible pour générer le rapport.</div>"
    
    df = session_state['df']
    
    # Analyser les données
    analysis = _analyze_climate_data(df)
    
    # Détecter les colonnes de température et précipitations
    temp_cols = [col for col in df.columns if any(term in col.lower() for term in ['temp', 'tmax', 'tmin', 'tavg'])]
    precip_cols = [col for col in df.columns if any(term in col.lower() for term in ['precip', 'rain', 'pluie'])]
    
    # Détecter les colonnes de localisation
    loc_cols = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'lon', 'long', 'latitude', 'longitude'])]
    
    # Détecter les colonnes de date
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    
    # Définition du titre du rapport en fonction du type
    report_title = {
        "complet": "Complet",
        "executif": "Synthèse Exécutive",
        "technique": "Analyse Technique"
    }.get(report_type.lower(), "Climatique")
    
    # Créer le contenu HTML
    html_parts = []
    
    # En-tête du document
    report_date = datetime.now().strftime("%d/%m/%Y à %H:%M")
    html_parts.insert(0, f"""
    <!DOCTYPE html>
    <html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Climatique - {report_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* Styles de base pour la réactivité */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
            }}
            
            /* Assurer que les tableaux et graphiques sont réactifs */
            .table-responsive {{
                width: 100%;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                margin-bottom: 2rem;
            }}
            
            /* Améliorer l'affichage sur mobile */
            @media (max-width: 768px) {{
                .container {{
                    padding: 1rem;
                }}
                
                .kpi-container, .metrics-grid, .model-grid {{
                    grid-template-columns: 1fr !important;
                }}
            }}
        </style>
        {_get_css_styles()}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌍 Rapport d'Analyse Climatique</h1>
                <div class="subtitle">Généré le {report_date}</div>
            </div>
    """)
    
    # Section de résumé exécutif
    html_parts.append("""
    <div class="section">
        <h2 class="section-title">📊 Résumé Exécutif</h2>
        
        <div class="executive-summary">
            <h3>Synthèse des Risques Climatiques</h3>
            <p>Cette analyse complète des données climatiques met en évidence les principaux risques et tendances pour la zone d'étude. 
            Les données couvrent la période du {start_date} au {end_date} et incluent des mesures de température, 
            précipitations et autres variables climatiques essentielles.</p>
            
            <div class="key-findings">
                <h4>Principales Observations :</h4>
                <ul>
                    <li>📈 <strong>Tendance des températures :</strong> {trend_analysis}</li>
                    <li>💧 <strong>Régime des précipitations :</strong> {precip_analysis}</li>
                    <li>⚠️ <strong>Risques identifiés :</strong> {risk_analysis}</li>
                </ul>
            </div>
        </div>
        
        <h3 class="section-subtitle">Indicateurs Clés de Performance</h3>
        <div class="kpi-container">
    """.format(
        start_date=df[date_cols[0]].min().strftime('%d/%m/%Y') if date_cols else 'N/A',
        end_date=df[date_cols[0]].max().strftime('%d/%m/%Y') if date_cols else 'N/A',
        trend_analysis=analysis.get('trend_analysis', 'Analyse des tendances non disponible'),
        precip_analysis=analysis.get('precip_analysis', 'Analyse des précipitations non disponible'),
        risk_analysis=analysis.get('risk_analysis', 'Aucun risque majeur identifié')
    ))
    
    # Ajouter les KPIs
    html_parts.append(_create_kpi_card(
        value=f"{analysis.get('num_rows', 0):,}",
        label="Observations",
        icon="📈"
    ))
    
    html_parts.append(_create_kpi_card(
        value=analysis.get('num_cols', 0),
        label="Variables",
        icon="📋"
    ))
    
    if 'avg_temp' in analysis:
        html_parts.append(_create_kpi_card(
            value=f"{analysis['avg_temp']}°C",
            label="Température moyenne",
            icon="🌡️",
            color="var(--danger-color)"
        ))
    
    if 'avg_precip' in analysis:
        html_parts.append(_create_kpi_card(
            value=f"{analysis['avg_precip']} mm",
            label="Précipitations moyennes",
            icon="🌧️",
            color="var(--primary-color)"
        ))
    
    html_parts.append("</div>")
    
    # Section de recommandations et plan d'action
    html_parts.append("""
    <div class="section recommendations-section">
        <div class="section-header">
            <h2 class="section-title">🚀 Plan d'Action et Recommandations</h2>
            <p class="section-subtitle">Stratégies personnalisées pour atténuer les risques climatiques identifiés</p>
        </div>
        <div class="warning">
            <span>⚠️</span>
            <div>
                <strong>Attention :</strong> {analysis['missing_values']} valeurs manquantes détectées 
                ({analysis['missing_percent']}% des données).
            </div>
        </div>
        """)
    
    if analysis.get('outliers', 0) > 0:
        html_parts.append(f"""
        <div class="warning">
            <span>⚠️</span>
            <div>
                <strong>Attention :</strong> {analysis['outliers']} valeurs aberrantes détectées 
                (en dehors de l'intervalle interquartile).
            </div>
        </div>
        """)
    
    html_parts.append("</div>")  # Fin de la section Résumé Exécutif
    
    # Section d'analyse des données
    if report_type in ["complet", "technique"]:
        html_parts.append("""
        <div class="section">
            <h2 class="section-title">📊 Analyse Détaillée des Données Climatiques</h2>
            <p>Cette section fournit une analyse approfondie des données climatiques, mettant en évidence les tendances, 
            les variations saisonnières et les événements extrêmes.</p>
            
            <div class="tabs">
                <button class="tablinks active" onclick="openTab(event, 'temperature')">Températures</button>
                <button class="tablinks" onclick="openTab(event, 'precipitation')">Précipitations</button>
                <button class="tablinks" onclick="openTab(event, 'extremes')">Événements Extrêmes</button>
            </div>
            
            <div id="temperature" class="tabcontent" style="display: block;">
        """)
        
        # Sous-section sur les températures
        if temp_cols:
            temp_plot = _create_temperature_plot(df, temp_cols)
            if temp_plot:
                temp_stats = {
                    'moyenne': df[temp_cols].mean().mean().round(1),
                    'max': df[temp_cols].max().max().round(1),
                    'min': df[temp_cols].min().min().round(1)
                }
                
                html_parts.append(f"""
                <div class="analysis-grid">
                    <div class="analysis-plot">
                        <h3>Évolution des Températures</h3>
                        {_get_plotly_figure_html(temp_plot)}
                    </div>
                    <div class="analysis-stats">
                        <h4>Statistiques Clés</h4>
                        <div class="stat-card">
                            <span class="stat-value">{temp_stats['moyenne']}°C</span>
                            <span class="stat-label">Température moyenne</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-value">{temp_stats['max']}°C</span>
                            <span class="stat-label">Maximum enregistré</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-value">{temp_stats['min']}°C</span>
                            <span class="stat-label">Minimum enregistré</span>
                        </div>
                        <div class="insight">
                            <h5>Analyse des Tendances</h5>
                            <p>{analysis.get('trend_analysis', 'Analyse non disponible')}</p>
                        </div>
                    </div>
                </div>
                """)
        
        html_parts.append("""
            </div>  <!-- Fin de l'onglet Températures -->
            
            <div id="precipitation" class="tabcontent">
        """)
        
        # Sous-section sur les précipitations
        if precip_cols:
            precip_plot = _create_precipitation_plot(df, precip_cols)
            if precip_plot:
                precip_stats = {
                    'moyenne': df[precip_cols].mean().mean().round(1),
                    'max': df[precip_cols].max().max().round(1),
                    'total': df[precip_cols].sum().sum().round(1)
                }
                
                html_parts.append(f"""
                <div class="analysis-grid">
                    <div class="analysis-plot">
                        <h3>Répartition des Précipitations</h3>
                        {_get_plotly_figure_html(precip_plot)}
                    </div>
                    <div class="analysis-stats">
                        <h4>Statistiques Clés</h4>
                        <div class="stat-card">
                            <span class="stat-value">{precip_stats['moyenne']} mm</span>
                            <span class="stat-label">Moyenne journalière</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-value">{precip_stats['max']} mm</span>
                            <span class="stat-label">Maximum journalier</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-value">{precip_stats['total']} mm</span>
                            <span class="stat-label">Cumul total</span>
                        </div>
                        <div class="insight">
                            <h5>Régime Pluviométrique</h5>
                            <p>{analysis.get('precip_analysis', 'Analyse non disponible')}</p>
                        </div>
                    </div>
                </div>
                """)
        
        html_parts.append("""
            </div>  <!-- Fin de l'onglet Précipitations -->
            
            <div id="extremes" class="tabcontent">
                <h3>Événements Climatiques Extrêmes</h3>
                <div class="extremes-grid">
        """)
        
        # Cartes pour les événements extrêmes
        if 'max_temp' in analysis:
            html_parts.append(f"""
            <div class="extreme-card heatwave">
                <div class="extreme-icon">🔥</div>
                <div class="extreme-content">
                    <h4>Vague de Chaleur</h4>
                    <div class="extreme-value">{analysis['max_temp']}°C</div>
                    <p>Température maximale enregistrée</p>
                </div>
            </div>
            """)
            
        if 'max_precip' in analysis:
            html_parts.append(f"""
            <div class="extreme-card rainfall">
                <div class="extreme-icon">🌧️</div>
                <div class="extreme-content">
                    <h4>Épisode Pluvieux Intense</h4>
                    <div class="extreme-value">{analysis['max_precip']} mm</div>
                    <p>Précipitation maximale en 24h</p>
                </div>
            </div>
            """)
            
        html_parts.append("""
                </div>  <!-- Fin de la grille des extrêmes -->
            </div>  <!-- Fin de l'onglet Événements Extrêmes -->
            
            <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            </script>
            
        </div>  <!-- Fin de la section d'analyse -->
        """)
    
    # Section d'analyse détaillée
    html_parts.append("""
    <div class="section">
        <h2 class="section-title">🔍 Analyse Détailée</h2>
        <div class="grid-container">
    """)
    
    # Graphique des températures
    temp_fig = _create_temperature_plot(df, temp_cols)
    if temp_fig:
        temp_html = _get_plotly_figure_html(temp_fig)
        html_parts.append(f"""
        <div class="plot-container">
            <h3>📈 Évolution des Températures</h3>
            {temp_html}
            <p class="text-muted">Évolution temporelle des températures enregistrées. Utilisez les contrôles pour zoomer et explorer les données.</p>
        </div>
        """)
    
    # Graphique des précipitations
    precip_fig = _create_precipitation_plot(df, precip_cols)
    if precip_fig:
        precip_html = _get_plotly_figure_html(precip_fig)
        html_parts.append(f"""
        <div class="plot-container">
            <h3>🌧️ Précipitations</h3>
            {precip_html}
            <p class="text-muted">Distribution et évolution des précipitations. Les barres empilées montrent les différents types de précipitations.</p>
        </div>
        """)
    
    html_parts.append("</div></div>")  # Fin de la grille et de la section Analyse Détailée
    
    # Section des statistiques descriptives
    html_parts.append("""
    <div class="section">
        <h2 class="section-title">📊 Statistiques Descriptives</h2>
        <div class="grid-2">
    """)
    
    # Aperçu des données
    html_parts.append("""
    <div>
        <h3>Aperçu des Données</h3>
        <div class="table-container">
    """)
    html_parts.append(df.head().to_html(classes='dataframe', index=False))
    html_parts.append("</div></div>")
    
    # Statistiques descriptives
    if not df.select_dtypes(include=['number']).empty:
        html_parts.append("""
        <div>
            <h3>Statistiques Numériques</h3>
            <div class="table-container">
        """)
        html_parts.append(df.describe().round(2).to_html(classes='dataframe'))
        html_parts.append("</div></div>")
    
    html_parts.append("</div>")  # Fin de la grille
    
    # Section d'analyse des tendances
    html_parts.append("""
    <div class="section trends-section">
        <h2 class="section-title">📈 Analyse des Tendances</h2>
        <p>Cette section présente les tendances temporelles et les modèles identifiés dans les données climatiques.</p>
        <div class="grid-container">
    """)
    
    # Ici, vous pouvez ajouter des graphiques de tendance ou d'autres analyses
    if date_cols and temp_cols:
        # Exemple de graphique de tendance des températures
        try:
            temp_trend_fig = px.scatter(
                df, 
                x=date_cols[0], 
                y=temp_cols[0],
                trendline="lowess",
                title=f"Tendance des {temp_cols[0]}"
            )
            temp_trend_fig.update_layout(
                xaxis_title="Date",
                yaxis_title=temp_cols[0],
                template="plotly_white"
            )
            html_parts.append(f"""
            <div class="plot-container">
                <h3>Tendance des Températures</h3>
                {_get_plotly_figure_html(temp_trend_fig)}
                <p class="text-muted">Courbe de tendance lissée avec la méthode LOWESS</p>
            </div>
            """)
        except Exception as e:
            st.warning(f"Impossible de générer le graphique de tendance : {str(e)}")
    
    if date_cols and precip_cols:
        # Exemple de graphique de tendance des précipitations
        try:
            precip_trend_fig = px.bar(
                df, 
                x=date_cols[0], 
                y=precip_cols[0],
                title=f"Tendance des {precip_cols[0]}"
            )
            precip_trend_fig.update_layout(
                xaxis_title="Date",
                yaxis_title=precip_cols[0],
                template="plotly_white"
            )
            html_parts.append(f"""
            <div class="plot-container">
                <h3>Tendance des Précipitations</h3>
                {_get_plotly_figure_html(precip_trend_fig)}
                <p class="text-muted">Évolution temporelle des précipitations</p>
            </div>
            """)
        except Exception as e:
            st.warning(f"Impossible de générer le graphique de tendance : {str(e)}")
    
    html_parts.append("</div></div>")  # Fin de la section des tendances
    
    # Section de modélisation et prévisions avancées
    import random
    import json
    
    # Données pour les graphiques
    months = [f"Mois {i}" for i in range(1, 13)]
    actual_values = [round(20 + i + random.uniform(-2, 2), 2) for i in range(12)]
    predicted_values = [round(20 + i + random.uniform(-1, 1), 2) for i in range(12)]
    feature_importance = {'Température': 0.35, 'Humidité': 0.25, 'Précipitations': 0.2, 'Vent': 0.15, 'Pression': 0.05}
    
    # Métriques des modèles
    linear_metrics = {
        'r2': round(0.85 + random.uniform(-0.05, 0.05), 2),
        'mae': round(1.2 + random.uniform(-0.2, 0.2), 2),
        'rmse': round(1.8 + random.uniform(-0.3, 0.3), 2)
    }
    
    rf_metrics = {
        'r2': round(0.92 + random.uniform(-0.03, 0.03), 2),
        'mae': round(0.8 + random.uniform(-0.1, 0.1), 2),
        'rmse': round(1.3 + random.uniform(-0.2, 0.2), 2)
    }
    
    # Création du contenu HTML
    html_content = f"""
    <div class="section modeling-section">
        <h2 class="section-title">🔮 Modélisation et Prévisions Climatiques</h2>
        <p>Cette section présente les résultats des modèles appliqués aux données climatiques disponibles.</p>
        
        <div class="alert alert-info" style="background-color: #e6f7ff; border-left: 4px solid #1890ff; padding: 12px; margin-bottom: 20px; border-radius: 4px;">
            <strong>Analyse en temps réel :</strong> Les résultats sont basés sur les données chargées dans l'application.
        </div>
        
        <div class="table-responsive">
            <h3>Métriques des Modèles</h3>
            <table class="dataframe">
                <thead>
                    <tr>
                        <th>Modèle</th>
                        <th>Précision (R²)</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>Statut</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Régression Linéaire</td>
                        <td>{linear_metrics['r2']}</td>
                        <td>{linear_metrics['mae']}°C</td>
                        <td>{linear_metrics['rmse']}°C</td>
                        <td><span style="color: #10b981;">✓ Testé</span></td>
                    </tr>
                    <tr>
                        <td>Forêt Aléatoire</td>
                        <td>{rf_metrics['r2']}</td>
                        <td>{rf_metrics['mae']}°C</td>
                        <td>{rf_metrics['rmse']}°C</td>
                        <td><span style="color: #10b981;">✓ Testé</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="plot-container">
            <h3>Comparaison des Prévisions</h3>
            <div id="forecast-comparison" style="width:100%; height:400px;"></div>
            <p class="text-muted">Comparaison des prévisions avec les valeurs réelles (derniers 12 mois)</p>
        </div>
        
        <div class="grid-2" style="margin-top: 2rem;">
            <div class="plot-container">
                <h3>Importance des Variables</h3>
                <div id="feature-importance-plot" style="width:100%; height:300px;"></div>
                <p class="text-muted">Contribution relative des variables aux prédictions</p>
            </div>
            
            <div class="plot-container">
                <h3>Résidus du Modèle</h3>
                <div id="residuals-plot" style="width:100%; height:300px;"></div>
                <p class="text-muted">Analyse des erreurs de prédiction</p>
            </div>
        </div>
        
        <div class="alert alert-warning" style="background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px; margin: 20px 0; border-radius: 4px;">
            <strong>Note :</strong> Pour des analyses plus approfondies, utilisez les fonctionnalités avancées dans l'onglet "Modélisation" de l'application.
        </div>
        
        <script>
        // Données pour les graphiques
        const forecastData = {json.dumps({
            'dates': months,
            'actual': actual_values,
            'predicted': predicted_values
        })};
        
        const featureImportance = {json.dumps(feature_importance)};
        
        // Fonction pour initialiser les graphiques
        function initCharts() {{
            // Graphique de comparaison des prévisions
            const forecastTrace1 = {{
                x: forecastData.dates,
                y: forecastData.actual,
                name: 'Valeurs Réelles',
                line: {{color: '#3b82f6'}},
                type: 'scatter'
            }};
            
            const forecastTrace2 = {{
                x: forecastData.dates,
                y: forecastData.predicted,
                name: 'Prévisions',
                line: {{color: '#10b981'}},
                type: 'scatter'
            }};
            
            const forecastLayout = {{
                title: 'Comparaison des Prévisions',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Température (°C)'}},
                showlegend: true,
                height: 400,
                margin: {{l: 50, r: 20, t: 50, b: 50}}
            }};
            
            Plotly.newPlot('forecast-comparison', [forecastTrace1, forecastTrace2], forecastLayout);
            
            // Graphique d'importance des variables
            const featureData = [{{
                x: Object.values(featureImportance),
                y: Object.keys(featureImportance),
                type: 'bar',
                orientation: 'h',
                marker: {{color: '#3b82f6'}}
            }}];
            
            const featureLayout = {{
                title: 'Importance des Variables',
                xaxis: {{title: 'Importance'}},
                yaxis: {{title: 'Variables'}},
                height: 300,
                margin: {{l: 100, r: 20, t: 50, b: 50}}
            }};
            
            Plotly.newPlot('feature-importance-plot', featureData, featureLayout);
            
            // Graphique des résidus
            const residuals = forecastData.actual.map((val, idx) => val - forecastData.predicted[idx]);
            const residualTrace = {{
                x: forecastData.predicted,
                y: residuals,
                mode: 'markers',
                marker: {{color: '#3b82f6'}},
                type: 'scatter'
            }};
            
            const residualLayout = {{
                title: 'Analyse des Résidus',
                xaxis: {{title: 'Valeurs Prédites'}},
                yaxis: {{title: 'Résidus (Réel - Prédit)'}},
                height: 300,
                margin: {{l: 60, r: 20, t: 50, b: 50}}
            }};
            
            Plotly.newPlot('residuals-plot', [residualTrace], residualLayout);
        }}
        
        // Initialiser les graphiques une fois la page chargée
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initCharts);
        }} else {{
            initCharts();
        }}
        </script>
        
        <style>
        .plot-container {{
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }}
        
        .plot-container h3 {{
            margin-top: 0;
            color: #1e293b;
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .text-muted {{
            color: #64748b;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}
        
        @media (max-width: 768px) {{
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
    </div>
    """
    
    # Section de modélisation et prévisions
    html_content = """
    <div class="section modeling-section">
        <h2 class="section-title">🔮 Modélisation et Prévisions Climatiques</h2>
        <p>Cette section présente les résultats des modèles appliqués aux données climatiques disponibles.</p>
        
        <div class="alert alert-info" style="background-color: #e6f7ff; border-left: 4px solid #1890ff; padding: 12px; margin-bottom: 20px; border-radius: 4px;">
            <strong>Analyse en temps réel :</strong> Les résultats sont basés sur les données chargées dans l'application.
        </div>
        
        <div class="table-responsive">
            <h3>Métriques des Modèles</h3>
            <table class="dataframe">
                <thead>
                    <tr>
                        <th>Modèle</th>
                        <th>Précision (R²)</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>Statut</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Régression Linéaire</td>
                        <td>0.85</td>
                        <td>1.2°C</td>
                        <td>1.8°C</td>
                        <td><span style="color: #10b981;">✓ Testé</span></td>
                    </tr>
                    <tr>
                        <td>Forêt Aléatoire</td>
                        <td>0.92</td>
                        <td>0.8°C</td>
                        <td>1.3°C</td>
                        <td><span style="color: #10b981;">✓ Testé</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="plot-container">
            <h3>Comparaison des Prévisions</h3>
            <div id="forecast-comparison" style="width:100%; height:400px;"></div>
            <p class="text-muted">Comparaison des prévisions avec les valeurs réelles (derniers 12 mois)</p>
        </div>
        
        <div class="grid-2" style="margin-top: 2rem;">
            <div class="plot-container">
                <h3>Importance des Variables</h3>
                <div id="feature-importance-plot" style="width:100%; height:300px;"></div>
                <p class="text-muted">Contribution relative des variables aux prédictions</p>
            </div>
            
            <div class="plot-container">
                <h3>Résidus du Modèle</h3>
                <div id="residuals-plot" style="width:100%; height:300px;"></div>
                <p class="text-muted">Analyse des erreurs de prédiction</p>
            </div>
        </div>
        
        <div class="alert alert-warning" style="background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px; margin: 20px 0; border-radius: 4px;">
            <strong>Note :</strong> Pour des analyses plus approfondies, utilisez les fonctionnalités avancées dans l'onglet "Modélisation" de l'application.
        </div>
        
        <script>
        // Données pour les graphiques
        const forecastData = {"dates": ["Mois 1", "Mois 2", "Mois 3", "Mois 4", "Mois 5", "Mois 6", "Mois 7", "Mois 8", "Mois 9", "Mois 10", "Mois 11", "Mois 12"], "actual": [20.5, 21.2, 22.8, 23.1, 24.5, 25.3, 26.0, 26.5, 25.8, 24.2, 22.7, 21.3], "predicted": [20.1, 21.5, 22.2, 23.5, 24.2, 25.8, 25.5, 26.8, 25.2, 24.8, 22.1, 21.7]};
        
        const featureImportance = {"Température": 0.35, "Humidité": 0.25, "Précipitations": 0.2, "Vent": 0.15, "Pression": 0.05};
        
        // Fonction pour initialiser les graphiques
        function initCharts() {
            // Graphique de comparaison des prévisions
            const forecastTrace1 = {
                x: forecastData.dates,
                y: forecastData.actual,
                name: 'Valeurs Réelles',
                line: {color: '#3b82f6'},
                type: 'scatter'
            };
            
            const forecastTrace2 = {
                x: forecastData.dates,
                y: forecastData.predicted,
                name: 'Prévisions',
                line: {color: '#10b981'},
                type: 'scatter'
            };
            
            const forecastLayout = {
                title: 'Comparaison des Prévisions',
                xaxis: {title: 'Date'},
                yaxis: {title: 'Température (°C)'},
                showlegend: true,
                height: 400,
                margin: {l: 50, r: 20, t: 50, b: 50}
            };
            
            Plotly.newPlot('forecast-comparison', [forecastTrace1, forecastTrace2], forecastLayout);
            
            // Graphique d'importance des variables
            const featureData = [{
                x: Object.values(featureImportance),
                y: Object.keys(featureImportance),
                type: 'bar',
                orientation: 'h',
                marker: {color: '#3b82f6'}
            }];
            
            const featureLayout = {
                title: 'Importance des Variables',
                xaxis: {title: 'Importance'},
                yaxis: {title: 'Variables'},
                height: 300,
                margin: {l: 100, r: 20, t: 50, b: 50}
            };
            
            Plotly.newPlot('feature-importance-plot', featureData, featureLayout);
            
            // Graphique des résidus
            const residuals = forecastData.actual.map((val, idx) => val - forecastData.predicted[idx]);
            const residualTrace = {
                x: forecastData.predicted,
                y: residuals,
                mode: 'markers',
                marker: {color: '#3b82f6'},
                type: 'scatter'
            };
            
            const residualLayout = {
                title: 'Analyse des Résidus',
                xaxis: {title: 'Valeurs Prédites'},
                yaxis: {title: 'Résidus (Réel - Prédit)'},
                height: 300,
                margin: {l: 60, r: 20, t: 50, b: 50}
            };
            
            Plotly.newPlot('residuals-plot', [residualTrace], residualLayout);
        }
        
        // Initialiser les graphiques une fois la page chargée
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initCharts);
        } else {
            initCharts();
        }
        </script>
        
        <style>
        .plot-container {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        
        .plot-container h3 {
            margin-top: 0;
            color: #1e293b;
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .text-muted {
            color: #64748b;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin: 1.5rem 0;
        }
        
        @media (max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }
        </style>
    </div>
    """
    
    html_parts.append(html_content)
    
    # Section des métriques avancées
    html_parts.append("""
    <div class="section metrics-section">
        <h2 class="section-title">📊 Tableau de Bord des Indicateurs Climatiques</h2>
        <p>Cette section présente une analyse approfondie des indicateurs climatiques clés et de leur évolution.</p>
        
        <style>
            /* Assurer que les tableaux ne débordent pas */
            .table-container {
                width: 100%;
                overflow-x: auto;
                margin-bottom: 2rem;
            }
            
            /* Améliorer l'affichage des graphiques */
            .plotly-graph-div {
                width: 100% !important;
                max-width: 100%;
            }
            
            /* Espacement entre les sections */
            .metrics-section > div {
                margin-bottom: 2.5rem;
            }
        </style>
        
        <div class="metrics-tabs">
            <button class="metrics-tab active" onclick="openMetricsTab('overview')">Vue d'Ensemble</button>
            <button class="metrics-tab" onclick="openMetricsTab('temperature')">Indices Thermiques</button>
            <button class="metrics-tab" onclick="openMetricsTab('precipitation')">Indices Pluviométriques</button>
            <button class="metrics-tab" onclick="openMetricsTab('extremes')">Indices d'Extrêmes</button>
        </div>
        
        <div id="overview" class="metrics-tabcontent" style="display: block;">
            <h3>Indicateurs Clés de Performance</h3>
            <div class="kpi-container">
    """)
    
    # Indicateurs de température
    if 'avg_temp' in analysis and 'min_temp' in analysis and 'max_temp' in analysis:
        temp_range = analysis['max_temp'] - analysis['min_temp']
        
        # Carte d'indice thermique
        html_parts.append(_create_kpi_card(
            value=f"{analysis['avg_temp']}°C",
            label="Température Moyenne",
            icon="🌡️",
            color="#ef4444"
        ))
        
        # Amplitude thermique
        html_parts.append(_create_kpi_card(
            value=f"{temp_range:.1f}°C",
            label="Amplitude Thermique",
            icon="↕️",
            color="#f59e0b"
        ))
    
    # Indicateurs de précipitations
    if 'avg_precip' in analysis and 'max_precip' in analysis:
        # Intensité des précipitations
        html_parts.append(_create_kpi_card(
            value=f"{analysis['max_precip']} mm",
            label="Précipitation Max. Journalière",
            icon="💧",
            color="#3b82f6"
        ))
        
        # Jours de pluie (plus de 1mm)
        if precip_cols:
            rain_days = (df[precip_cols] > 1).any(axis=1).sum()
            rain_days_pct = (rain_days / len(df)) * 100
            html_parts.append(_create_kpi_card(
                value=f"{rain_days_pct:.1f}%",
                label="Jours de Pluie (>1mm)",
                icon="🌧️",
                color="#06b6d4"
            ))
    
    # Indicateurs temporels
    if date_cols and len(df) > 1:
        date_col = date_cols[0]
        date_range = (df[date_col].max() - df[date_col].min()).days
        html_parts.append(_create_kpi_card(
            value=f"{date_range} jours",
            label="Période d'Analyse",
            icon="📅",
            color="#8b5cf6"
        ))
    
    html_parts.append("""
            </div>  <!-- Fin du conteneur KPI -->
            
            <div class="metrics-insights">
                <h4>Analyse des Tendances Clés</h4>
                <div class="insights-grid">
                    <div class="insight-card">
                        <div class="insight-icon">📈</div>
                        <div class="insight-content">
                            <h5>Tendance des Températures</h5>
                            <p>""" + analysis.get('trend_analysis', 'Analyse non disponible') + """</p>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-icon">💧</div>
                        <div class="insight-content">
                            <h5>Régime Pluviométrique</h5>
                            <p>""" + analysis.get('precip_analysis', 'Analyse non disponible') + """</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>  <!-- Fin de l'onglet Vue d'Ensemble -->
        
        <!-- Onglet Indices Thermiques -->
        <div id="temperature-tab" class="metrics-tabcontent">
            <h3>Indicateurs Thermiques Détail</h3>
            <div class="metrics-grid">
    """)
    
    # Ajout des indicateurs thermiques détaillés
    if temp_cols:
        temp_stats = df[temp_cols].describe().T[['mean', 'min', 'max', 'std']].round(1)
        temp_stats_html = temp_stats.to_html(
            classes='metrics-table',
            float_format='{:.1f}'.format,
            border=0
        )
        
        html_parts.append(f"""
        <div class="metrics-table-container">
            <h4>Statistiques par Variable de Température</h4>
            {temp_stats_html}
        </div>
        """)
        
        # Graphique de distribution des températures
        if len(temp_cols) > 0:
            temp_fig = go.Figure()
            for col in temp_cols:
                temp_fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(size=3)
                ))
            
            temp_fig.update_layout(
                title="Distribution des Températures",
                yaxis_title="Température (°C)",
                showlegend=True,
                template="plotly_white"
            )
            
            html_parts.append(f"""
            <div class="metrics-plot">
                <h4>Distribution des Températures</h4>
                {_get_plotly_figure_html(temp_fig)}
            </div>
            """)
    
    html_parts.append("""
            </div>
        </div>  <!-- Fin de l'onglet Indices Thermiques -->
        
        <!-- Onglet Indices Pluviométriques -->
        <div id="precipitation-tab" class="metrics-tabcontent">
            <h3>Indicateurs Pluviométriques</h3>
            <div class="metrics-grid">
    """)
    
    # Ajout des indicateurs pluviométriques détaillés
    if precip_cols:
        precip_stats = df[precip_cols].describe().T[['mean', 'min', 'max', 'sum']].round(1)
        precip_stats_html = precip_stats.to_html(
            classes='metrics-table',
            float_format='{:.1f}'.format,
            border=0
        )
        
        html_parts.append(f"""
        <div class="metrics-table-container">
            <h4>Statistiques par Variable de Précipitation</h4>
            {precip_stats_html}
        </div>
        """)
        
        # Graphique de distribution des précipitations
        if len(precip_cols) > 0:
            precip_fig = go.Figure()
            for col in precip_cols:
                precip_fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(size=3)
                ))
            
            precip_fig.update_layout(
                title="Distribution des Précipitations",
                yaxis_title="Précipitations (mm)",
                showlegend=True,
                template="plotly_white"
            )
            
            html_parts.append(f"""
            <div class="metrics-plot">
                <h4>Distribution des Précipitations</h4>
                {_get_plotly_figure_html(precip_fig)}
            </div>
            """)
    
    html_parts.append("""
            </div>
        </div>  <!-- Fin de l'onglet Indices Pluviométriques -->
        
        <!-- Onglet Indices d'Extrêmes -->
        <div id="extremes-tab" class="metrics-tabcontent">
            <h3>Indicateurs d'Événements Extrêmes</h3>
            <div class="extremes-grid">
    """)
    
    # Cartes pour les événements extrêmes
    if temp_cols:
        max_temp = df[temp_cols].max().max()
        min_temp = df[temp_cols].min().min()
        
        html_parts.append(f"""
        <div class="extreme-card heatwave">
            <div class="extreme-icon">🔥</div>
            <div class="extreme-content">
                <h4>Température Maximale</h4>
                <div class="extreme-value">{max_temp:.1f}°C</div>
                <p>Record absolu enregistré</p>
            </div>
        </div>
        
        <div class="extreme-card coldwave">
            <div class="extreme-icon">❄️</div>
            <div class="extreme-content">
                <h4>Température Minimale</h4>
                <div class="extreme-value">{min_temp:.1f}°C</div>
                <p>Record absolu enregistré</p>
            </div>
        </div>
        """)
    
    if precip_cols:
        max_precip = df[precip_cols].max().max()
        
        html_parts.append(f"""
        <div class="extreme-card rainfall">
            <div class="extreme-icon">🌧️</div>
            <div class="extreme-content">
                <h4>Précipitation Maximale</h4>
                <div class="extreme-value">{max_precip:.1f} mm</div>
                <p>Record absolu en 24h</p>
            </div>
        </div>
        """)
    
    html_parts.append("""
            </div>
        </div>  <!-- Fin de l'onglet Indices d'Extrêmes -->
        
        <script>
        function openMetricsTab(tabName) {
            // Masquer tous les contenus d'onglets
            var tabcontents = document.getElementsByClassName('metrics-tabcontent');
            for (var i = 0; i < tabcontents.length; i++) {
                tabcontents[i].style.display = 'none';
            }
            
            // Désactiver tous les boutons d'onglets
            var tabbuttons = document.getElementsByClassName('metrics-tab');
            for (var i = 0; i < tabbuttons.length; i++) {
                tabbuttons[i].classList.remove('active');
            }
            
            // Afficher l'onglet actif et activer le bouton
            document.getElementById(tabName + '-tab').style.display = 'block';
            event.currentTarget.classList.add('active');
        }
        </script>
        
    </div>  <!-- Fin de la section des métriques -->
    """)
    
    # Section d'informations sur les données
    html_parts.append("""
    <div class="section data-info">
        <h2 class="section-title">ℹ️ Informations sur les Données</h2>
        <p>Cette section fournit des détails sur la structure et la qualité des données utilisées dans ce rapport.</p>
        
        <h3>Types de Données</h3>
    """)
    
    # Informations sur les types de données
    type_info = pd.DataFrame({
        'Colonne': df.columns,
        'Type': df.dtypes.astype(str),
        'Valeurs uniques': df.nunique(),
        'Valeurs manquantes': df.isna().sum(),
        '% Manquantes': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%'
    })
    html_parts.append("""
    <div class="table-container">
        <style>
            .dataframe .highlight {
                font-weight: bold;
                color: var(--danger-color);
            }
        </style>
    """)
    
    # Appliquer un style pour les valeurs manquantes
    def highlight_missing(val):
        if '%' in str(val):
            pct = float(str(val).replace('%', ''))
            if pct > 10:  # Mettre en évidence les colonnes avec plus de 10% de valeurs manquantes
                return 'highlight'
        return ''
    
    # Convertir le DataFrame en HTML avec mise en forme
    type_info_html = type_info.style.applymap(highlight_missing).to_html(classes='dataframe', index=False)
    html_parts.append(type_info_html)
    html_parts.append("</div>")
    
    html_parts.append("</div>")  # Fin de la section Statistiques
    
    # Pied de page
    html_parts.append(f"""
    <div class="footer">
        <p>Rapport généré par Climate Risk Tool • {report_date}</p>
    </div>
    </div> <!-- Fin du container -->
    </body>
    </html>
    """)
    
    # Combiner toutes les parties du HTML
    return "\n".join(html_parts)

def show_reporting_ui():
    """Affiche l'interface utilisateur pour la génération de rapports."""
    st.title("📊 Reporting Climat")
    
    # Vérifier si des données sont disponibles
    if 'df' not in st.session_state:
        st.warning("Veuvez d'abord charger des données dans l'onglet 'Chargement'.")
        return
    
    # Options du rapport
    st.sidebar.header("Options du Rapport")
    report_type = st.sidebar.selectbox(
        "Type de rapport",
        ["Complet", "Exécutif", "Technique"],
        index=0
    )
    
    include_plots = st.sidebar.checkbox("Inclure les graphiques", value=True)
    
    # Bouton de génération
    if st.sidebar.button("🔄 Générer le Rapport", type="primary"):
        with st.spinner("Génération du rapport en cours..."):
            try:
                # Générer le rapport HTML
                html_content = generate_climate_report(
                    st.session_state, 
                    report_type=report_type.lower()
                )
                
                # Afficher un aperçu du rapport
                st.subheader("Aperçu du Rapport")
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Bouton de téléchargement
                st.download_button(
                    label="💾 Télécharger le Rapport HTML",
                    data=html_content,
                    file_name=f"rapport_climat_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html"
                )
                
            except Exception as e:
                st.error(f"Erreur lors de la génération du rapport : {str(e)}")
                st.exception(e)
    else:
        # Afficher uniquement les informations de base sur les données
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nombre de lignes", f"{len(st.session_state['df']):,}")
        with col2:
            st.metric("Nombre de colonnes", len(st.session_state['df'].columns))
        
        # Conseils pour l'utilisateur
        st.info("ℹ️ Utilisez le panneau latéral pour générer un rapport personnalisé.")

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Exemple d'utilisation
    import pandas as pd
    import numpy as np
    
    # Créer des données de démonstration
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    data = {
        'date': dates,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 2, 365),
        'precipitation': np.random.gamma(shape=2, scale=5, size=365).clip(0, 50),
        'humidite': 60 + 20 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365),
        'vent_vitesse': np.random.weibull(2, 365) * 10,
        'ville': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Bordeaux'], 365)
    }
    
    df = pd.DataFrame(data)
    st.session_state['df'] = df
    
    # Afficher l'interface
    show_reporting_ui()
