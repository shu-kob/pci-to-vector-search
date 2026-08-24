"""Why Analysis & Explainability Engine (Powered by Gemini / Vertex AI).

Explains 'WHY' two users are similar beyond surface-level purchase logs (What).
Transforms raw vector similarity into actionable consumer insights.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel


class WhyAnalysisEngine:
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "asia-northeast1",
        model_name: str = "gemini-3.7-flash",
    ):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
        self.location = location
        self.model_name = model_name
        self._init_client()



    def _init_client(self) -> None:
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            self.available = True
        except Exception as e:
            print(f"Warning: Gemini init failed ({e}). Fallback to rule-based why analysis.")
            self.model = None
            self.available = False

    def explain_why_similar(
        self,
        target_user: Dict[str, Any],
        matched_user: Dict[str, Any],
        search_mode: str = "why_vector",  # 'what_classic' or 'why_vector'
    ) -> Dict[str, str]:
        """Generates concise 'Why' insights explaining the underlying motivation and consumer context."""
        if not self.available or self.model is None:
            return self._fallback_explanation(target_user, matched_user, search_mode)

        prompt = f"""
あなたは顧客インサイトと購買行動分析の専門マーケターです。
以下の2名のユーザ情報を比較し、「なぜこの2人が似ているのか（行動の裏にあるWhy / 生活動機・価値観の共通点）」と「マーケティング施策への示唆」を日本語で簡潔に分析してください。

【対象ユーザA】
- 属性: 年齢層={target_user.get('age_band')}, 性別={target_user.get('gender')}, 地域={target_user.get('area')}
- ペルソナ/セグメント: {target_user.get('segment_name')}
- プロフィール文 (Why): {target_user.get('profile_text')}
- 関心事: {', '.join(target_user.get('interests', []))}

【類似と判定されたユーザB】
- 属性: 年齢層={matched_user.get('age_band')}, 性別={matched_user.get('gender')}, 地域={matched_user.get('area')}
- ペルソナ/セグメント: {matched_user.get('segment_name')}
- プロフィール文 (Why): {matched_user.get('profile_text')}
- 関心事: {', '.join(matched_user.get('interests', []))}

【探索手法】
{search_mode} (What=過去の購買・評価アイテムの一致による探索 / Why=プロフィールや生活価値観のベクトルによる探索)

以下のJSON形式で回答してください:
{{
  "why_common_motivation": "2人に共通する根底の生活価値観や購買動機（1〜2文）",
  "what_vs_why_gap": "購買アイテム(What)と生活動機(Why)の観点から見える特徴や注意点（1〜2文）",
  "marketing_action": "この2人に有効なアプローチや推薦施策（1文）"
}}
"""
        try:
            config = GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=500,
            )
            response = self.model.generate_content(prompt, generation_config=config)
            return json.loads(response.text)
        except Exception as e:
            print(f"Gemini API generation error: {e}. Using fallback.")
            return self._fallback_explanation(target_user, matched_user, search_mode)

    def _fallback_explanation(
        self,
        target_user: Dict[str, Any],
        matched_user: Dict[str, Any],
        search_mode: str,
    ) -> Dict[str, str]:
        t_seg = target_user.get("segment_name", "不明")
        m_seg = matched_user.get("segment_name", "不明")
        same_seg = t_seg == m_seg

        if same_seg:
            return {
                "why_common_motivation": f"両者ともに「{t_seg}」としての生活様式を持ち、日々の生活における関心事や優先事項（{', '.join(target_user.get('interests', [])[:3])}など）の文脈が高度に一致しています。",
                "what_vs_why_gap": "購入アイテムが完全一致していなくても、根底にある価値観が同じであるため、潜在的なニーズや共感ポイントが共通しています。",
                "marketing_action": f"{t_seg}向けの実用性や世界観に訴求したパーソナライズド提案が効果的です。",
            }
        else:
            return {
                "why_common_motivation": f"セグメントは「{t_seg}」と「{m_seg}」で異なりますが、生活スタイルやツール選びにおいて部分的な動機の一致が見られます。",
                "what_vs_why_gap": "What（評価アイテム）の見かけの一致または局所的な共通項により引き当てられていますが、生活全体のWhy（動機）にはズレが存在する可能性があります。",
                "marketing_action": "アイテム単体ではなく、それぞれの生活文脈に合わせたメッセージングのチューニングが必要です。",
            }
