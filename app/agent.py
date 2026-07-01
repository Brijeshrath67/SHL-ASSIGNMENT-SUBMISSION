from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from .schemas import ChatMessage, ChatResponse, Recommendation
from .retrieval import CatalogRetriever

logger = logging.getLogger(__name__)

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

MAX_TURNS = 8
MAX_RECOMMENDATIONS = 10

OFF_TOPIC_KEYWORDS = {
    "weather", "sports", "politics", "cooking", "recipe", "movie", "music",
    "game", "play", "travel", "vacation", "news", "stock", "crypto",
    "legal advice", "medical", "health advice", "investment",
}

LEGAL_REQUIRED_PATTERNS = [
    r"\blegal(ly)?\s+(requir|obligation|implic|ramif|question|advice|opinion|must)",
    r"\b(am\s+I|are\s+we|is\s+it|do\s+I|does\s+this)\s+.*\b(legal|required|obligated|compliant)\b",
    r"\brequire.*by\s+(law|regulation|statute)\b",
    r"\bis\s+this\s+(legal|lawful|compliant|allow)",
    r"\b(legal|law).*test.*(satisfy|require|mandat)",
    r"\bsatisfy\s+(legal|regulatory|compliance)\s+requirement",
]

HIRING_CONTEXT_TERMS = {
    "hire", "assessment", "test", "candidate", "recruit",
    "job", "role", "position", "skill", "competenc", "evaluate",
    "screen", "shortlist", "applicant", "talent", "shl",
    "solution", "assess", "personality", "cognitive", "aptitude",
    "simulation", "interview", "knowledge", "ability",
    "leadership", "senior", "manager", "engineer", "developer",
    "contact center", "customer service", "sales", "graduate",
    "entry level", "executive", "director", "pool",
}

ROLE_TERMS = [
    "software engineer", "cashier", "sales", "manager", "developer",
    "analyst", "nurse", "doctor", "teacher", "customer service",
    "driver", "technician", "associate", "representative", "clerk",
    "assistant", "supervisor", "director", "cxo", "executive",
    "graduate", "intern", "leadership", "contact center", "call center",
    "candidate", "agent", "senior",
]

SKILL_TERMS = [
    "personality", "cognitive", "ability", "aptitude", "skill",
    "knowledge", "technical", "programming", "java", "python",
    "excel", "word", "accounting", "finance", "communication",
    "teamwork", "problem.solv", "reasoning", "motivation",
    "situational", "simulation", "interview",
]

JOB_LEVEL_TERMS = [
    "entry.level", "graduate", "manager", "executive", "director",
    "cxo", "senior", "professional", "intern",
]


class Agent:

    def __init__(self, retriever: CatalogRetriever):
        self.retriever = retriever

    async def process(
        self, messages: list[ChatMessage]
    ) -> ChatResponse:
        history = self._format_history(messages)
        last_msg = messages[-1].content if messages else ""
        turn_count = (len(messages) + 1) // 2

        if turn_count > MAX_TURNS:
            return ChatResponse(
                reply="This conversation has reached its maximum length. Please start a new conversation if you need further assistance.",
                recommendations=[],
                end_of_conversation=True,
            )

        if self._is_off_topic(last_msg):
            return self._refuse()

        intent = self._classify_intent(last_msg, turn_count)
        logger.info("Intent: %s (turn %d)", intent, turn_count)

        if intent == "farewell":
            return self._farewell()

        if intent == "compare":
            return await self._handle_compare(last_msg, history, messages)

        if intent == "refine":
            return await self._handle_refine(last_msg, messages)

        if intent == "clarify":
            return self._handle_clarify(last_msg, turn_count, history, messages)

        return await self._handle_recommend(last_msg, history, messages)

    def _build_search_query(self, messages: list[ChatMessage]) -> str:
        user_parts = []
        for m in messages:
            if m.role == "user":
                user_parts.append(m.content)
        return " ".join(user_parts)

    def _format_history(self, messages: list[ChatMessage]) -> str:
        lines = []
        for m in messages:
            role = "Candidate" if m.role == "user" else "Consultant"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def _is_off_topic(self, text: str) -> bool:
        lower = text.lower()
        if any(re.search(p, lower) for p in LEGAL_REQUIRED_PATTERNS):
            return True
        if any(kw in lower for kw in OFF_TOPIC_KEYWORDS):
            if not any(t in lower for t in HIRING_CONTEXT_TERMS):
                return True
        return False

    def _classify_intent(self, text: str, turn_count: int) -> str:
        lower = text.lower()

        farewell_patterns = [
            r"\b(thanks|thank you|bye|goodbye|that'?s all|that'?s it|all set|done)\b",
            r"\b(no more|nothing else|that helps|clear)\b",
            r"(perfect|confirmed).*?(that'?s|that is|this is) (what we need|all)",
            r"\bperfect\b.*\b(that'?s|that is)\b",
            r"\bthat'?s good\b",
            r"\blocking it in\b",
            r"\bfinal list\b",
            r"keep.*as.is",
            r"\bkeep it\b",
            r"\bcancel\b",
            r"\bconfirmed\b",
            r"\bthat covers it\b",
        ]
        for pat in farewell_patterns:
            if re.search(pat, lower):
                return "farewell"

        compare_patterns = [
            r"\b(compare|difference|vs|versus|which (is )?better|between)\b",
        ]
        for pat in compare_patterns:
            if re.search(pat, lower):
                names = self._find_named_products(lower)
                if len(names) >= 2:
                    return "compare"

        refine_patterns = [
            r"\b(add|include|also).{0,30}\b(test|assessment)\b",
            r"\b(add|include|also)\b",
            r"\b(drop|remove|replace)\b",
        ]
        for pat in refine_patterns:
            if re.search(pat, lower):
                return "refine"

        has_role = bool(re.search("|".join(ROLE_TERMS), lower))
        has_skill = bool(re.search("|".join(SKILL_TERMS), lower))
        has_level = bool(re.search("|".join(JOB_LEVEL_TERMS), lower))
        has_context = has_role or has_skill or has_level

        if not has_context:
            pure_greeting = re.match(
                r"^(hi|hello|hey|help)$", lower
            )
            if pure_greeting:
                return "clarify"
            word_count = len(lower.split())
            if word_count < 3:
                return "clarify"

        return "recommend"

    def _find_named_products(self, text: str) -> list[str]:
        found = []
        for item in self.retriever.catalog:
            name = item.get("name", "")
            if name.lower() in text.lower():
                found.append(name)
        return list(set(found))

    def _refuse(self) -> ChatResponse:
        return ChatResponse(
            reply=(
                "I specialize in SHL assessment recommendations for hiring and "
                "talent management. I can't help with that request. "
                "Please let me know what role or skills you're hiring for!"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    def _farewell(self) -> ChatResponse:
        return ChatResponse(
            reply="You're welcome! If you need further assistance with SHL assessments in the future, feel free to reach out.",
            recommendations=[],
            end_of_conversation=True,
        )

    def _handle_clarify(self, text: str, turn_count: int, history: str, messages: list[ChatMessage]) -> ChatResponse:
        lower = text.lower().strip()
        word_count = len(lower.split())

        if word_count <= 3 and turn_count > 1:
            all_user_text = " ".join(
                m.content for m in messages if m.role == "user"
            ).lower()
            contact_center = any(
                w in all_user_text for w in ["contact center", "call center", "agent"]
            )
            language_mentioned = any(
                w in lower for w in ["english", "spanish", "french", "bilingual"]
            )
            if language_mentioned and contact_center:
                reply = (
                    "Thanks! For the contact center assessments, do you need a specific "
                    "accent or dialect? We have options for US, UK, Australian, and Indian "
                    "English, as well as Spanish and bilingual variants."
                )
                return ChatResponse(
                    reply=reply, recommendations=[], end_of_conversation=False
                )
            location_mentioned = any(
                w in lower for w in ["us", "uk", "india", "australia", "canada", "europe"]
            )
            if location_mentioned:
                from .retrieval import LOCATION_QUERY_MAP
                sub_q = LOCATION_QUERY_MAP.get(lower, text)
                query = self._build_search_query(messages[:-1]) + " " + sub_q if len(messages) > 1 else sub_q
                results = self.retriever.search(query, k=15)
                if not results:
                    results = self.retriever.search(text, k=10)
                items = results[:MAX_RECOMMENDATIONS]
                if items:
                    reply = (
                        f"Based on your requirements, here are some relevant "
                        f"SHL assessments:\n\n"
                        f"{self._format_items(items[:8])}\n"
                        f"Would you like more details on any of these?"
                    )
                    return ChatResponse(
                        reply=reply,
                        recommendations=[
                            Recommendation(
                                name=item.get("name", ""),
                                url=item.get("url", ""),
                                test_type=item.get("test_type", []),
                            )
                            for item in items[:8]
                        ],
                        end_of_conversation=False,
                    )
                else:
                    return ChatResponse(
                        reply="Thanks. Let me search for assessments matching your criteria.",
                        recommendations=[],
                        end_of_conversation=False,
                    )

        if word_count <= 2:
            reply = (
                "Thanks for that. Are there any specific skills or competencies "
                "you're looking to assess for this role?"
            )
            return ChatResponse(
                reply=reply, recommendations=[], end_of_conversation=False
            )

        reply = (
            "I'd be happy to help find the right SHL assessments for your hiring needs! "
            "Could you tell me about the role you're hiring for? "
            "For example, what job level (entry, graduate, manager, executive) "
            "and what key skills or competencies are you looking to assess?"
        )
        return ChatResponse(
            reply=reply, recommendations=[], end_of_conversation=False
        )

    async def _handle_compare(
        self, text: str, history: str, messages: list[ChatMessage]
    ) -> ChatResponse:
        product_names = self._find_named_products(text)
        items = [
            p
            for p in self.retriever.catalog
            if p.get("name", "") in product_names
        ]
        if len(items) < 2:
            query = self._build_search_query(messages)
            results = self.retriever.search(query, k=5)
            items = results[:2]

        if not items:
            return ChatResponse(
                reply="I couldn't find those specific products in my catalog. Could you clarify which SHL products you'd like me to compare?",
                recommendations=[],
                end_of_conversation=False,
            )

        if LLM_API_KEY:
            return await self._llm_response(
                text, history, items, messages, intent="compare"
            )
        return self._template_compare(items)

    def _template_compare(self, items: list[dict]) -> ChatResponse:
        lines = ["Here's a comparison of the assessments you asked about:\n"]
        for item in items:
            name = item.get("name", "Unknown")
            desc = item.get("description", "No description available.")
            keys = ", ".join(item.get("test_type", [])) or "N/A"
            levels = ", ".join(item.get("job_levels", [])) or "N/A"
            duration = item.get("duration")
            dur_str = f"{duration} min" if duration else "N/A"
            lines.append(
                f"**{name}**\n"
                f"- Description: {desc[:200]}\n"
                f"- Test Type(s): {keys}\n"
                f"- Job Level(s): {levels}\n"
                f"- Duration: {dur_str}\n"
            )
        lines.append(
            "Would you like more details on any of these, or would you like me "
            "to recommend a combination that fits your needs?"
        )
        return ChatResponse(
            reply="\n".join(lines),
            recommendations=[
                Recommendation(
                    name=item.get("name", ""),
                    url=item.get("url", ""),
                    test_type=item.get("test_type", []),
                )
                for item in items
            ],
            end_of_conversation=False,
        )

    def _assess_info_completeness(
        self, messages: list[ChatMessage]
    ) -> tuple[bool, str | None]:
        all_user_text = " ".join(
            m.content for m in messages if m.role == "user"
        ).lower()
        has_role = bool(re.search("|".join(ROLE_TERMS), all_user_text))
        has_skill = bool(re.search("|".join(SKILL_TERMS), all_user_text))
        has_level = bool(re.search("|".join(JOB_LEVEL_TERMS), all_user_text))

        if not has_level:
            return False, "What job level are you hiring for (entry, graduate, manager, or executive)?"
        if not has_skill and not has_role:
            return False, "What specific skills or competencies are you looking to assess?"
        return True, None

    def _needs_follow_up(
        self, messages: list[ChatMessage], items: list[dict]
    ) -> str | None:
        turn_count = (len(messages) + 1) // 2
        if turn_count > 1:
            return None
        _, question = self._assess_info_completeness(messages)
        if question:
            return question
        return None

    async def _handle_refine(
        self, text: str, messages: list[ChatMessage]
    ) -> ChatResponse:
        lower = text.lower()
        existing = self._extract_previous_recommendations(messages)
        existing_names = {r.name for r in existing}

        add_terms = ["add", "include", "also"]
        drop_terms = ["drop", "remove", "replace"]
        query = self._build_search_query(messages)

        if any(re.search(rf"\b{t}\b", lower) for t in drop_terms):
            drop_match = re.search(
                r"\b(drop|remove)\s+(?:the\s+)?(.+)", lower
            )
            if drop_match:
                target = drop_match.group(2).strip()
                for item_name in list(existing_names):
                    if item_name.lower().startswith(target.lower()[:3]):
                        existing_names.discard(item_name)
            replace_match = re.search(
                r"replace\s+(?:it\s+)?with\s+(.+)", lower, re.IGNORECASE
            )
            if not replace_match:
                replace_match = re.search(
                    r"replace\s+(.+?)\s+with\s+(.+)", lower, re.IGNORECASE
                )

        add_results = []
        if any(re.search(rf"\b{t}\b", lower) for t in add_terms):
            add_query = query
            specific_match = re.search(
                r"\b(add|include)\s+(?:a\s+|an\s+|the\s+)?(.+)", lower
            )
            if specific_match:
                add_query = specific_match.group(2).strip()
            add_results = self.retriever.search(add_query, k=5)

        final_items = []
        for item in self.retriever.catalog:
            if item.get("name", "") in existing_names:
                final_items.append(item)

        for item in add_results:
            if item.get("name", "") not in existing_names:
                final_items.append(item)

        if not final_items:
            final_items = add_results or self.retriever.search(query, k=10)

        top = final_items[:MAX_RECOMMENDATIONS]

        if LLM_API_KEY:
            return await self._llm_response(
                text, self._format_history(messages), top, messages, intent="recommend"
            )
        return self._template_recommend(top, messages)

    async def _handle_recommend(
        self, text: str, history: str, messages: list[ChatMessage]
    ) -> ChatResponse:
        query = self._build_search_query(messages)
        results = self.retriever.search(query, k=15)
        existing = self._extract_previous_recommendations(messages)
        combined = self._merge_recommendations(existing, results)

        if not results:
            return ChatResponse(
                reply=(
                    "I'm not finding any SHL assessments that match your criteria. "
                    "Could you provide more details about the role or skills you're "
                    "looking to assess?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        top = combined[:MAX_RECOMMENDATIONS]

        if LLM_API_KEY:
            return await self._llm_response(
                text, history, top, messages, intent="recommend"
            )
        return self._template_recommend(top, messages)

    def _template_recommend(
        self, items: list[dict], messages: list[ChatMessage]
    ) -> ChatResponse:
        if not items:
            return ChatResponse(
                reply="I couldn't find matching assessments. Could you share more about your hiring needs?",
                recommendations=[],
                end_of_conversation=False,
            )

        follow_up = self._needs_follow_up(messages, items)
        turn_count = (len(messages) + 1) // 2
        recommend_count = min(len(items), 5 if turn_count <= 2 else 8)

        if follow_up and turn_count <= 2:
            reply = (
                f"Based on what you've shared, here are some "
                f"potentially relevant SHL assessments:\n\n"
                f"{self._format_items(items[:recommend_count])}\n\n"
                f"{follow_up}"
            )
        else:
            reply = (
                f"Based on your requirements, here are some relevant "
                f"SHL assessments:\n\n"
                f"{self._format_items(items[:recommend_count])}\n"
                f"Would you like more details on any of these, or would you like to "
                f"refine the list further?"
            )
        return ChatResponse(
            reply=reply,
            recommendations=[
                Recommendation(
                    name=item.get("name", ""),
                    url=item.get("url", ""),
                    test_type=item.get("test_type", []),
                )
                for item in items[:recommend_count]
            ],
            end_of_conversation=False,
        )

    def _format_items(self, items: list[dict]) -> str:
        lines = []
        for i, item in enumerate(items, 1):
            name = item.get("name", "Unknown")
            desc = item.get("description", "")[:120]
            test_type = ", ".join(item.get("test_type", [])) or "Assessment"
            lines.append(
                f"{i}. **{name}** ({test_type})\n"
                f"   {desc}"
            )
        return "\n\n".join(lines)

    def _extract_previous_recommendations(
        self, messages: list[ChatMessage]
    ) -> list[Recommendation]:
        recs = []
        for msg in messages:
            if msg.role == "assistant":
                for item in self.retriever.catalog:
                    if item.get("name", "") in msg.content:
                        recs.append(
                            Recommendation(
                                name=item["name"],
                                url=item.get("url", ""),
                                test_type=item.get("test_type", []),
                            )
                        )
        seen = set()
        unique = []
        for r in recs:
            if r.name not in seen:
                seen.add(r.name)
                unique.append(r)
        return unique

    def _merge_recommendations(
        self,
        existing: list[Recommendation],
        results: list[dict],
    ) -> list[dict]:
        existing_names = {r.name for r in existing}
        merged = []
        for item in results:
            if item.get("name", "") in existing_names:
                merged.append(item)
        for item in results:
            if item.get("name", "") not in existing_names:
                merged.append(item)
        return merged

    async def _llm_response(
        self,
        text: str,
        history: str,
        items: list[dict],
        messages: list[ChatMessage],
        intent: str,
    ) -> ChatResponse:
        try:
            return await self._call_llm(text, history, items, messages, intent)
        except Exception as exc:
            logger.warning("LLM call failed, using template fallback: %s", exc)
            if intent == "compare":
                return self._template_compare(items)
            return self._template_recommend(items, messages)

    async def _call_llm(
        self,
        text: str,
        history: str,
        items: list[dict],
        messages: list[ChatMessage],
        intent: str,
    ) -> ChatResponse:
        catalog_block = self._format_catalog_for_prompt(items)
        query = self._build_search_query(messages)

        system_prompt = (
            "You are an expert SHL assessment consultant. Your role is to help "
            "hiring managers select the right SHL assessments.\n\n"
            "AVAILABLE PRODUCTS (use ONLY these):\n"
            f"{catalog_block}\n\n"
            "BEHAVIOR RULES:\n"
            "1. If the user is vague, ask clarifying questions about the role, skills, and job level.\n"
            "2. If enough information is available, recommend 1-10 products from the catalog above.\n"
            "3. If comparing, give a structured comparison with differences in test type, duration, and job levels.\n"
            "4. If refining, update the previous recommendation with new products.\n"
            "5. NEVER recommend products not in the catalog above.\n"
            "6. NEVER make up product details.\n"
            "7. Politely refuse off-topic questions.\n"
            "8. End the conversation only when the user explicitly confirms or says goodbye.\n\n"
            "RESPONSE FORMAT (use exactly these markers):\n"
            "##REPLY\n"
            "[your conversational reply]\n"
            "##RECOMMENDATIONS\n"
            "[comma-separated product names from the catalog, or 'NONE']\n"
            "##END\n"
            "[true/false]"
        )

        user_prompt = (
            "Conversation so far:\n"
            f"{history}\n\n"
            f"Latest user message: {text}\n"
            f"Search query used: {query}\n\n"
            "Generate the next response following the format above."
        )

        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]

        return self._parse_llm_response(raw, items)

    def _format_catalog_for_prompt(self, items: list[dict]) -> str:
        lines = []
        for item in items:
            name = item.get("name", "")
            desc = item.get("description", "")[:150]
            test_type = ", ".join(item.get("test_type", [])) or "N/A"
            levels = ", ".join(item.get("job_levels", [])) or "N/A"
            duration = item.get("duration")
            dur_str = f"{duration} min" if duration else "N/A"
            lines.append(
                f"- {name}: {desc} | Type: {test_type} | Levels: {levels} | Duration: {dur_str}"
            )
        return "\n".join(lines)

    def _parse_llm_response(
        self, raw: str, items: list[dict]
    ) -> ChatResponse:
        reply_match = re.search(
            r"##REPLY\s*\n(.*?)(?=\n##RECOMMENDATIONS|\Z)", raw, re.DOTALL
        )
        reply = reply_match.group(1).strip() if reply_match else raw.strip()

        rec_match = re.search(
            r"##RECOMMENDATIONS\s*\n(.*?)(?=\n##END|\Z)", raw, re.DOTALL
        )
        rec_text = rec_match.group(1).strip() if rec_match else ""
        rec_names = (
            [n.strip() for n in rec_text.split(",") if n.strip()]
            if rec_text and rec_text.upper() != "NONE"
            else []
        )
        recommendations = []
        for name in rec_names:
            match = next(
                (
                    item
                    for item in items
                    if item.get("name", "").lower() == name.lower()
                ),
                None,
            )
            if match:
                recommendations.append(
                    Recommendation(
                        name=match.get("name", ""),
                        url=match.get("url", ""),
                        test_type=match.get("test_type", []),
                    )
                )

        end_match = re.search(r"##END\s*\n(.+)", raw, re.DOTALL)
        end_val = end_match.group(1).strip().lower() if end_match else "false"
        end_of_conversation = end_val == "true"

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation,
        )
