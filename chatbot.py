import json
import os
import re
import uuid
from pathlib import Path
from urllib import error, request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CITY_TIER = {
    "mumbai": "metro",
    "delhi": "metro",
    "new delhi": "metro",
    "bengaluru": "metro",
    "bangalore": "metro",
    "hyderabad": "metro",
    "chennai": "metro",
    "kolkata": "metro",
    "pune": "tier1",
    "ahmedabad": "tier1",
    "surat": "tier1",
    "lucknow": "tier2",
    "jaipur": "tier2",
    "nagpur": "tier2",
    "indore": "tier2",
    "bhopal": "tier2",
    "noida": "tier2",
    "gurgaon": "tier2",
    "gurugram": "tier2",
    "chandigarh": "tier2",
    "kochi": "tier2",
    "cochin": "tier2",
    "coimbatore": "tier2",
    "vadodara": "tier2",
    "patna": "tier2",
    "visakhapatnam": "tier2",
    "vizag": "tier2",
}


FIELD_ORDER = [
    "project_type",
    "city",
    "property_area_sqft",
    "rooms",
    "scope",
    "materials_grade",
    "furniture_included",
    "estimated_labour_days",
    "contractor_type",
    "design_style",
    "n_materials",
]


FIELD_LABELS = {
    "project_type": "project type",
    "city": "city",
    "property_area_sqft": "carpet area",
    "rooms": "number of rooms",
    "scope": "scope of work",
    "materials_grade": "material grade",
    "furniture_included": "furniture inclusion",
    "estimated_labour_days": "estimated labour days",
    "contractor_type": "execution partner",
    "design_style": "design style",
    "n_materials": "number of material types",
}


CHOICES = {
    "project_type": ("residential", "office", "retail", "commercial"),
    "scope": ("decor only", "partial", "full"),
    "materials_grade": ("economy", "standard", "premium", "luxury"),
    "contractor_type": ("contractor", "designer-led", "agency", "owner"),
    "design_style": (
        "modern",
        "contemporary",
        "minimalist",
        "traditional",
        "industrial",
        "colonial",
        "scandinavian",
    ),
}


GRADE_ORDER = ("economy", "standard", "premium", "luxury")

PROJECT_TYPE_ALIASES = {
    "home": "residential",
    "house": "residential",
    "flat": "residential",
    "apartment": "residential",
    "villa": "residential",
    "2bhk": "residential",
    "3bhk": "residential",
    "4bhk": "residential",
    "workspace": "office",
    "shop": "retail",
    "store": "retail",
    "showroom": "retail",
}

SCOPE_ALIASES = {
    "decor": "decor only",
    "decor-only": "decor only",
    "decor only": "decor only",
    "partial": "partial",
    "full": "full",
    "complete": "full",
    "turnkey": "full",
    "end to end": "full",
    "end-to-end": "full",
}

CONTRACTOR_ALIASES = {
    "designer led": "designer-led",
    "designer-led": "designer-led",
    "interior designer": "designer-led",
    "design studio": "designer-led",
    "agency": "agency",
    "firm": "agency",
    "company": "agency",
    "owner managed": "owner",
    "owner-managed": "owner",
    "self managed": "owner",
    "self-managed": "owner",
    "self": "owner",
    "contractor": "contractor",
}

MATERIAL_WORDS = {
    "plywood",
    "laminate",
    "veneer",
    "marble",
    "granite",
    "quartz",
    "acrylic",
    "glass",
    "wood",
    "mdf",
    "hdf",
    "steel",
    "brass",
    "tile",
    "tiles",
    "wallpaper",
    "paint",
    "stone",
    "wpc",
    "pvc",
}


def format_inr(value):
    value = int(round(float(value)))
    if value >= 10_000_000:
        return f"Rs {value / 10_000_000:.2f} Cr"
    if value >= 100_000:
        return f"Rs {value / 100_000:.2f} L"
    return f"Rs {value:,}"


class RagStore:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.chunks = self._load_chunks()
        self.vectorizer = None
        self.matrix = None
        if self.chunks:
            texts = [chunk["text"] for chunk in self.chunks]
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(texts)

    def _load_chunks(self):
        chunks = [
            {
                "source": "chatbot-field-guide",
                "text": (
                    "The VastuCost chatbot must collect project_type, city, "
                    "property_area_sqft, rooms, scope, materials_grade, "
                    "furniture_included, estimated_labour_days, contractor_type, "
                    "design_style, and n_materials before producing an estimate. "
                    "It should also support natural-language extraction, budget "
                    "guidance, what-if comparisons, cost drivers, planning advice, "
                    "cost breakup, adaptive follow-up questions, and a project brief."
                ),
            },
            {
                "source": "city-tier-guide",
                "text": (
                    "City tiers: metro cities are Mumbai, Delhi, Bengaluru, "
                    "Hyderabad, Chennai, and Kolkata. Tier1 cities are Pune, "
                    "Ahmedabad, and Surat. Other supported cities are treated "
                    "as tier2 for pricing."
                ),
            },
            {
                "source": "model-input-guide",
                "text": (
                    "Material grades follow a cost order: economy, standard, "
                    "premium, luxury. Scope values are decor only, partial, and full. "
                    "The ML model predicts total_cost_inr and returns a low/high band. "
                    "Breakups and recommendations are planning heuristics layered on "
                    "top of the ML total, not independently trained component outputs."
                ),
            },
        ]

        file_sources = [
            self.project_root / "vastucost" / "README.md",
            self.project_root / "model" / "model_info.json",
        ]
        for path in file_sources:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix.lower() == ".json":
                try:
                    text = json.dumps(json.loads(text), indent=2)
                except json.JSONDecodeError:
                    pass
            chunks.extend(self._split_text(path.name, text))
        return chunks

    @staticmethod
    def _split_text(source, text, size=900, overlap=120):
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return []
        pieces = []
        start = 0
        while start < len(clean):
            pieces.append({"source": source, "text": clean[start : start + size]})
            start += size - overlap
        return pieces

    def search(self, query, top_k=3):
        if self.vectorizer is None or self.matrix is None or not query.strip():
            return self.chunks[:top_k]
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in ranked if scores[i] > 0][:top_k]


class XAIClient:
    def __init__(self):
        self.api_key = os.environ.get("XAI_API_KEY", "").strip()
        self.model = os.environ.get("XAI_MODEL", "grok-4-fast").strip()
        self.base_url = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")

    @property
    def enabled(self):
        return bool(self.api_key)

    def chat(self, messages, temperature=0.35, max_tokens=220):
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=18) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
            return None

        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        ) or None


class VastuCostChatbot:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.sessions = {}
        self.rag = RagStore(project_root)
        self.llm = XAIClient()
        self.model_info = self._load_model_info()

    def _load_model_info(self):
        info_path = self.project_root / "model" / "model_info.json"
        if not info_path.exists():
            return {}
        try:
            return json.loads(info_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _new_state():
        return {"answers": {}, "messages": [], "budget_inr": None}

    def reset(self, session_id=None):
        session_id = session_id or str(uuid.uuid4())
        self.sessions[session_id] = self._new_state()
        return session_id

    def handle(self, session_id, message, estimate_fn):
        session_id = session_id or self.reset()
        state = self.sessions.setdefault(session_id, self._new_state())
        state.setdefault("answers", {})
        state.setdefault("messages", [])
        state.setdefault("budget_inr", None)

        message = (message or "").strip()
        incoming_updates = {}

        if message:
            state["messages"].append({"role": "user", "content": message})
            budget = self._extract_budget(message)
            if budget:
                state["budget_inr"] = budget

            if self._has_required_answers(state["answers"]) and self._is_what_if_request(message):
                result = self._what_if_result(session_id, state, message, estimate_fn)
                state["messages"].append({"role": "assistant", "content": result["reply"]})
                return result

            expected = self._next_missing(state["answers"])
            incoming_updates = self._extract_answers(message, expected)
            state["answers"].update(incoming_updates)

        missing = self._next_missing(state["answers"])
        if missing:
            reply = self._question_reply(state, message, missing)
            state["messages"].append({"role": "assistant", "content": reply})
            return {
                "session_id": session_id,
                "phase": "collecting",
                "reply": reply,
                "next_field": missing,
                "answers": state["answers"],
                "budget_inr": state.get("budget_inr"),
                "llm_enabled": self.llm.enabled,
                "sources": [c["source"] for c in self.rag.search(message or missing)],
            }

        payload = self._prediction_payload(state["answers"])
        estimate = estimate_fn(payload)

        if message and self._is_project_brief_request(message) and not incoming_updates:
            reply = self._brief_reply(state, estimate)
        elif message and self._is_explain_request(message) and not incoming_updates:
            reply = self._explain_reply(state, estimate)
        else:
            reply = self._final_reply(state, estimate, estimate_fn)

        state["messages"].append({"role": "assistant", "content": reply})
        return {
            "session_id": session_id,
            "phase": "estimate",
            "reply": reply,
            "answers": state["answers"],
            "budget_inr": state.get("budget_inr"),
            "payload": payload,
            "estimate": estimate,
            "llm_enabled": self.llm.enabled,
            "sources": [c["source"] for c in self.rag.search("final estimate cost band")],
        }

    @staticmethod
    def _has_required_answers(answers):
        return all(field in answers for field in FIELD_ORDER)

    def _next_missing(self, answers):
        for field in FIELD_ORDER:
            if field not in answers:
                return field
        return None

    def _question_reply(self, state, message, missing):
        base = self._question_text(missing, state)
        collected = self._collected_summary(state["answers"])
        context = self._context(message or missing)
        llm_reply = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the VastuCost estimator chatbot. Ask exactly one "
                        "short follow-up question for the missing field. Be warm, "
                        "but do not invent values or produce an estimate until all "
                        "required fields are collected. Use the provided context only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Collected answers: {collected}\n"
                        f"Missing field: {FIELD_LABELS[missing]}\n"
                        f"Question to ask: {base}\n"
                        f"Relevant RAG context:\n{context}"
                    ),
                },
            ]
        )
        if llm_reply:
            return llm_reply
        if state["answers"]:
            return f"Got it. {base}"
        return (
            "Hi, I can estimate your interior project cost by asking a few quick "
            f"questions. {base}"
        )

    def _question_text(self, missing, state):
        answers = state["answers"]
        budget = state.get("budget_inr")

        if missing == "project_type":
            return "What type of space is this: residential, office, retail, or commercial?"
        if missing == "city":
            return "Which city is the project in?"
        if missing == "property_area_sqft":
            city = answers.get("city")
            prefix = f"For {city}, " if city else ""
            return f"{prefix}what is the carpet area in square feet?"
        if missing == "rooms":
            if answers.get("project_type") == "residential":
                return "How many rooms or BHK should be included?"
            return "How many major rooms or zones should be included?"
        if missing == "scope":
            area = answers.get("property_area_sqft")
            prefix = f"For {area:,} sqft, " if area else ""
            return f"{prefix}is the scope decor only, partial, or full interior?"
        if missing == "materials_grade":
            if budget:
                return (
                    f"With a budget of {format_inr(budget)}, which material grade "
                    "should I price: economy, standard, premium, or luxury?"
                )
            return "Which material grade should I price: economy, standard, premium, or luxury?"
        if missing == "furniture_included":
            return "Should modular furniture such as kitchen, wardrobes, or storage be included?"
        if missing == "estimated_labour_days":
            suggestion = self._suggest_labour_days(answers)
            if suggestion:
                return (
                    f"Roughly how many labour days should I assume? "
                    f"For this scope, around {suggestion} days is a reasonable starting point."
                )
            return "Roughly how many labour days should I assume?"
        if missing == "contractor_type":
            return "Who will execute it: contractor, designer-led team, agency, or owner-managed?"
        if missing == "design_style":
            project = answers.get("project_type", "project")
            return (
                f"Which style fits the {project}: modern, contemporary, minimalist, "
                "traditional, industrial, colonial, or scandinavian?"
            )
        if missing == "n_materials":
            return (
                "How many main material families will be used? For example, "
                "plywood, laminate, quartz, glass, and paint would be 5."
            )
        return f"What is the {FIELD_LABELS.get(missing, missing)}?"

    @staticmethod
    def _suggest_labour_days(answers):
        area = answers.get("property_area_sqft")
        scope = answers.get("scope")
        if not area or not scope:
            return None
        divisor = {"decor only": 65, "partial": 40, "full": 26}.get(scope, 40)
        return max(7, min(180, round(float(area) / divisor)))

    def _final_reply(self, state, estimate, estimate_fn):
        answers = state["answers"]
        lines = [
            (
                f"Estimate: {format_inr(estimate['total'])} "
                f"({format_inr(estimate['low'])} to {format_inr(estimate['high'])}), "
                f"about Rs {estimate['cost_per_sqft']:,}/sqft."
            ),
            f"Project brief: {self._project_brief(answers)}",
            "Likely cost drivers: " + "; ".join(self._cost_drivers(answers)),
            "Planning breakup: " + self._format_breakup(self._cost_breakup(answers, estimate)),
        ]

        budget_lines = self._budget_advice(answers, estimate, estimate_fn, state.get("budget_inr"))
        if budget_lines:
            lines.append("Budget check: " + " ".join(budget_lines))

        suggestions = self._planning_suggestions(
            answers, estimate, estimate_fn, state.get("budget_inr")
        )
        if suggestions:
            lines.append("Suggestions: " + " ".join(suggestions))

        lines.append(f"Model: {estimate['model']} with R2 {estimate['r2']}.")
        return "\n\n".join(lines)

    def _explain_reply(self, state, estimate):
        answers = state["answers"]
        features = self._model_feature_sentence()
        lines = [
            f"The estimate is {format_inr(estimate['total'])}, or Rs {estimate['cost_per_sqft']:,}/sqft.",
            "Main reasons: " + "; ".join(self._cost_drivers(answers)),
            "Planning breakup: " + self._format_breakup(self._cost_breakup(answers, estimate)),
        ]
        if features:
            lines.append(features)
        lines.append(
            "The low/high band is useful because vendor quotes can shift with finish brands, site access, custom carpentry, and city labour rates."
        )
        return "\n\n".join(lines)

    def _brief_reply(self, state, estimate):
        return (
            f"Project brief: {self._project_brief(state['answers'])}\n\n"
            f"Current ML estimate: {format_inr(estimate['total'])} "
            f"({format_inr(estimate['low'])} to {format_inr(estimate['high'])}), "
            f"about Rs {estimate['cost_per_sqft']:,}/sqft."
        )

    def _what_if_result(self, session_id, state, message, estimate_fn):
        variants = self._scenario_variants(message)
        base_payload = self._prediction_payload(state["answers"])
        base_estimate = estimate_fn(base_payload)

        if not variants:
            reply = (
                "I can compare that. Tell me the changed choice, for example "
                "\"what if we use standard materials\" or \"compare full vs partial\"."
            )
            return {
                "session_id": session_id,
                "phase": "comparison",
                "reply": reply,
                "answers": state["answers"],
                "estimate": base_estimate,
                "llm_enabled": self.llm.enabled,
                "sources": [c["source"] for c in self.rag.search(message)],
            }

        lines = [f"Current estimate: {format_inr(base_estimate['total'])}."]
        scenarios = []
        for label, updates in variants[:4]:
            if all(state["answers"].get(key) == value for key, value in updates.items()):
                continue
            scenario_answers = dict(state["answers"])
            scenario_answers.update(updates)
            scenario_payload = self._prediction_payload(scenario_answers)
            scenario_estimate = estimate_fn(scenario_payload)
            delta = scenario_estimate["total"] - base_estimate["total"]
            if delta > 0:
                delta_text = f"{format_inr(delta)} higher"
            elif delta < 0:
                delta_text = f"{format_inr(abs(delta))} lower"
            else:
                delta_text = "about the same"
            lines.append(
                f"{label}: {format_inr(scenario_estimate['total'])} "
                f"({delta_text}), range {format_inr(scenario_estimate['low'])} "
                f"to {format_inr(scenario_estimate['high'])}."
            )
            scenarios.append(
                {
                    "label": label,
                    "updates": updates,
                    "payload": scenario_payload,
                    "estimate": scenario_estimate,
                    "delta": delta,
                }
            )

        if not scenarios:
            reply = (
                f"That scenario matches the current saved inputs, so the estimate remains "
                f"{format_inr(base_estimate['total'])}."
            )
            return {
                "session_id": session_id,
                "phase": "comparison",
                "reply": reply,
                "answers": state["answers"],
                "budget_inr": state.get("budget_inr"),
                "estimate": base_estimate,
                "scenarios": scenarios,
                "llm_enabled": self.llm.enabled,
                "sources": [c["source"] for c in self.rag.search(message)],
            }

        lines.append(
            "I have not changed your saved answers; this is a side-by-side planning scenario."
        )
        reply = "\n\n".join(lines)
        return {
            "session_id": session_id,
            "phase": "comparison",
            "reply": reply,
            "answers": state["answers"],
            "budget_inr": state.get("budget_inr"),
            "estimate": base_estimate,
            "scenarios": scenarios,
            "llm_enabled": self.llm.enabled,
            "sources": [c["source"] for c in self.rag.search(message)],
        }

    def _context(self, query):
        chunks = self.rag.search(query, top_k=3)
        return "\n".join(f"- {c['source']}: {c['text']}" for c in chunks)

    @staticmethod
    def _collected_summary(answers):
        if not answers:
            return "none yet"
        return ", ".join(
            f"{FIELD_LABELS.get(k, k)}={v}"
            for k, v in answers.items()
            if k != "city_tier"
        )

    def _extract_answers(self, message, expected=None):
        text = self._clean_text(message)
        updates = {}

        for value in CHOICES["project_type"]:
            if re.search(rf"\b{re.escape(value)}\b", text):
                updates["project_type"] = value
                break
        if "project_type" not in updates:
            for alias, value in PROJECT_TYPE_ALIASES.items():
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    updates["project_type"] = value
                    break
        if "project_type" not in updates and re.search(r"\b\d\s*bhk\b", text):
            updates["project_type"] = "residential"

        for city, tier in sorted(CITY_TIER.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(city)}\b", text):
                updates["city"] = city.title()
                updates["city_tier"] = tier
                break
        if "metro" in text:
            updates.setdefault("city", "Metro")
            updates["city_tier"] = "metro"
        elif "tier 1" in text or "tier1" in text:
            updates.setdefault("city", "Tier1")
            updates["city_tier"] = "tier1"
        elif "tier 2" in text or "tier2" in text:
            updates.setdefault("city", "Tier2")
            updates["city_tier"] = "tier2"

        area_match = re.search(
            r"(\d{2,5}(?:,\d{2,3})*(?:\.\d+)?)\s*"
            r"(?:sq\.?\s*ft|sqft|sft|square\s*feet|square\s*ft)",
            text,
        )
        if area_match:
            updates["property_area_sqft"] = self._number(area_match.group(1))

        room_match = re.search(r"(\d{1,2})\s*(?:bhk|bedroom|bedrooms|room|rooms)", text)
        if room_match:
            updates["rooms"] = self._number(room_match.group(1))

        for alias, value in sorted(SCOPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                updates["scope"] = value
                break

        for value in CHOICES["materials_grade"]:
            if value in text:
                updates["materials_grade"] = value
                break
        if "high end" in text or "high-end" in text:
            updates["materials_grade"] = "luxury"
        elif "basic" in text or "budget grade" in text:
            updates["materials_grade"] = "economy"

        furniture_context = re.search(
            r"\b(furniture|modular|wardrobe|wardrobes|storage|kitchen)\b", text
        )
        if re.search(r"\b(no|not|without|exclude|excluded|skip)\b", text):
            if expected == "furniture_included" or furniture_context:
                updates["furniture_included"] = 0
        elif re.search(r"\b(yes|include|included|with|true|add)\b", text):
            if expected == "furniture_included" or furniture_context:
                updates["furniture_included"] = 1

        labour_match = re.search(r"(\d{1,4})\s*(?:labou?r\s*)?(?:day|days)", text)
        if labour_match:
            updates["estimated_labour_days"] = self._number(labour_match.group(1))
        else:
            week_match = re.search(r"(\d{1,2})\s*(?:week|weeks)", text)
            if week_match and ("labour" in text or "timeline" in text or expected == "estimated_labour_days"):
                updates["estimated_labour_days"] = self._number(week_match.group(1)) * 7

        for alias, value in sorted(CONTRACTOR_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                updates["contractor_type"] = value
                break

        for value in CHOICES["design_style"]:
            if value in text:
                updates["design_style"] = value
                break
        if "scandi" in text:
            updates["design_style"] = "scandinavian"

        material_count = re.search(r"(\d{1,2})\s*(?:main\s*)?(?:material|materials|finishes)", text)
        if material_count:
            updates["n_materials"] = self._number(material_count.group(1))
        else:
            found_materials = {
                material
                for material in MATERIAL_WORDS
                if re.search(rf"\b{re.escape(material)}\b", text)
            }
            if len(found_materials) >= 2:
                updates["n_materials"] = len(found_materials)

        if expected and expected not in updates:
            inferred = self._extract_expected(text, expected)
            if inferred is not None:
                updates[expected] = inferred
                if expected == "city":
                    updates["city_tier"] = CITY_TIER.get(str(inferred).lower(), "tier2")

        return self._valid_updates(updates)

    def _extract_expected(self, text, expected):
        if expected in {"property_area_sqft", "rooms", "estimated_labour_days", "n_materials"}:
            match = re.search(r"\d{1,5}(?:,\d{2,3})*", text)
            return self._number(match.group(0)) if match else None
        if expected == "furniture_included":
            if re.search(r"\b(no|not|without|exclude|excluded)\b", text):
                return 0
            if re.search(r"\b(yes|include|included|with|true)\b", text):
                return 1
        if expected == "city" and text.strip():
            city = text.strip().split(",")[0].title()
            return city
        if expected == "project_type":
            if re.search(r"\b\d\s*bhk\b", text):
                return "residential"
            for alias, value in PROJECT_TYPE_ALIASES.items():
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return value
        if expected == "scope":
            for alias, value in sorted(SCOPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return value
        if expected == "contractor_type":
            for alias, value in sorted(CONTRACTOR_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    return value
        if expected == "materials_grade":
            if "high end" in text or "high-end" in text:
                return "luxury"
            if "basic" in text or "budget grade" in text:
                return "economy"
        if expected == "design_style" and "scandi" in text:
            return "scandinavian"
        for value in CHOICES.get(expected, set()):
            if value in text:
                return value
        return None

    @staticmethod
    def _number(value):
        return int(float(str(value).replace(",", "")))

    @staticmethod
    def _valid_updates(updates):
        cleaned = {}
        for key, value in updates.items():
            if key == "property_area_sqft" and not (100 <= int(value) <= 10000):
                continue
            if key == "rooms" and not (1 <= int(value) <= 30):
                continue
            if key == "estimated_labour_days" and not (1 <= int(value) <= 1000):
                continue
            if key == "n_materials" and not (1 <= int(value) <= 20):
                continue
            if key == "city" and len(str(value)) > 60:
                continue
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _prediction_payload(answers):
        city_tier = answers.get("city_tier")
        if not city_tier:
            city_tier = CITY_TIER.get(str(answers.get("city", "")).lower(), "tier2")
        return {
            "project_type": answers["project_type"],
            "property_area_sqft": answers["property_area_sqft"],
            "rooms": answers["rooms"],
            "scope": answers["scope"],
            "materials_grade": answers["materials_grade"],
            "furniture_included": answers["furniture_included"],
            "estimated_labour_days": answers["estimated_labour_days"],
            "contractor_type": answers["contractor_type"],
            "design_style": answers["design_style"],
            "city_tier": city_tier,
            "n_materials": answers["n_materials"],
        }

    @staticmethod
    def _clean_text(message):
        return re.sub(r"\s+", " ", (message or "").strip().lower())

    def _extract_budget(self, message):
        text = self._clean_text(message)
        if not re.search(
            r"\b(budget|afford|spend|under|within|cap|limit|rs|inr|lakh|lakhs|lac|crore|cr)\b",
            text,
        ):
            return None

        patterns = [
            r"(?:rs|inr)\s*([0-9][0-9,]*(?:\.\d+)?)\s*(crore|cr|lakhs|lakh|lac|l|k)?",
            r"([0-9][0-9,]*(?:\.\d+)?)\s*(crore|cr|lakhs|lakh|lac|l|k)\b",
            r"\b(?:budget|afford|spend|under|within|cap|limit)\D{0,18}([0-9][0-9,]*(?:\.\d+)?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            amount = float(match.group(1).replace(",", ""))
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            if unit in {"crore", "cr"}:
                amount *= 10_000_000
            elif unit in {"lakhs", "lakh", "lac", "l"}:
                amount *= 100_000
            elif unit == "k":
                amount *= 1_000
            elif amount < 50_000:
                continue
            return int(round(amount))
        return None

    @staticmethod
    def _is_what_if_request(message):
        text = VastuCostChatbot._clean_text(message)
        return bool(
            re.search(
                r"\b(what if|compare|comparison|versus|vs|instead of|rather than|scenario|option)\b",
                text,
            )
        )

    @staticmethod
    def _is_explain_request(message):
        text = VastuCostChatbot._clean_text(message)
        return bool(
            re.search(
                r"\b(why|explain|driver|drivers|reason|high|low|expensive|costly|breakdown|break up|breakup)\b",
                text,
            )
        )

    @staticmethod
    def _is_project_brief_request(message):
        text = VastuCostChatbot._clean_text(message)
        return bool(
            re.search(
                r"\b(brief|summary|summarize|proposal|vendor|designer note|project note)\b",
                text,
            )
        )

    def _scenario_variants(self, message):
        text = self._clean_text(message)
        variants = []

        grade_hits = self._ordered_hits(text, GRADE_ORDER)
        if len(grade_hits) >= 2:
            for grade in grade_hits:
                variants.append((f"{grade.title()} materials", {"materials_grade": grade}))

        scope_hits = self._ordered_hits(text, CHOICES["scope"])
        if len(scope_hits) >= 2:
            for scope in scope_hits:
                variants.append((f"{scope.title()} scope", {"scope": scope}))

        if "furniture" in text and re.search(r"\b(vs|versus|compare|with|without)\b", text):
            if re.search(r"\bwith\b", text):
                variants.append(("With modular furniture", {"furniture_included": 1}))
            if "without" in text or "no furniture" in text:
                variants.append(("Without modular furniture", {"furniture_included": 0}))

        if variants:
            return self._dedupe_variants(variants)

        target = text
        for delimiter in ("instead of", "rather than", " over "):
            if delimiter in target:
                target = target.split(delimiter, 1)[0]
                break
        target = re.sub(r"\b(what if|compare|comparison|scenario|option|we|i|choose|use|switch|change|to)\b", " ", target)
        updates = self._extract_answers(target, None)
        updates = {k: v for k, v in updates.items() if k in FIELD_ORDER or k == "city_tier"}
        if not updates:
            return []
        return [(self._label_for_updates(updates), updates)]

    @staticmethod
    def _ordered_hits(text, options):
        hits = []
        for option in options:
            match = re.search(rf"\b{re.escape(option)}\b", text)
            if match:
                hits.append((match.start(), option))
        return [value for _, value in sorted(hits)]

    @staticmethod
    def _dedupe_variants(variants):
        seen = set()
        unique = []
        for label, updates in variants:
            key = tuple(sorted(updates.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append((label, updates))
        return unique

    @staticmethod
    def _label_for_updates(updates):
        labels = []
        if "materials_grade" in updates:
            labels.append(f"{str(updates['materials_grade']).title()} materials")
        if "scope" in updates:
            labels.append(f"{str(updates['scope']).title()} scope")
        if "furniture_included" in updates:
            labels.append("With furniture" if updates["furniture_included"] else "Without furniture")
        if "city" in updates:
            labels.append(f"{updates['city']} location")
        if "contractor_type" in updates:
            labels.append(f"{str(updates['contractor_type']).title()} execution")
        if "design_style" in updates:
            labels.append(f"{str(updates['design_style']).title()} style")
        return ", ".join(labels) or "Scenario"

    def _cost_drivers(self, answers):
        drivers = []
        area = float(answers.get("property_area_sqft", 0))
        rooms = int(answers.get("rooms", 0))
        grade = answers.get("materials_grade")
        scope = answers.get("scope")
        city_tier = answers.get("city_tier") or CITY_TIER.get(
            str(answers.get("city", "")).lower(), "tier2"
        )

        if area >= 1200:
            drivers.append(f"large {int(area):,} sqft area")
        elif area <= 550:
            drivers.append(f"compact {int(area):,} sqft area, where fixed costs weigh more per sqft")
        else:
            drivers.append(f"{int(area):,} sqft project size")

        if grade in {"premium", "luxury"}:
            drivers.append(f"{grade} material grade")
        elif grade:
            drivers.append(f"{grade} material grade keeps finishes controlled")

        if scope == "full":
            drivers.append("full-scope interior work")
        elif scope == "partial":
            drivers.append("partial scope limits the work package")
        elif scope == "decor only":
            drivers.append("decor-only scope keeps structural work low")

        if city_tier == "metro":
            drivers.append("metro city pricing")
        elif city_tier == "tier1":
            drivers.append("tier1 city pricing")

        if answers.get("furniture_included"):
            drivers.append("modular furniture included")
        if rooms >= 4:
            drivers.append(f"{rooms} rooms/zones to finish")

        return drivers[:5]

    def _model_feature_sentence(self):
        top_features = self.model_info.get("top_features") or []
        if not top_features:
            return ""
        readable = [
            self._readable_feature(item.get("feature", ""))
            for item in top_features[:4]
            if item.get("feature")
        ]
        if not readable:
            return ""
        return "The trained model's strongest signals include " + ", ".join(readable) + "."

    @staticmethod
    def _readable_feature(feature):
        return (
            feature.replace("_x_", " times ")
            .replace("_", " ")
            .replace("sqft", "sqft")
            .strip()
        )

    def _cost_breakup(self, answers, estimate):
        total = int(estimate["total"])
        scope = answers.get("scope")
        furniture = bool(answers.get("furniture_included"))

        if scope == "decor only":
            weights = [
                ("Decor and finishes", 0.34),
                ("Loose furniture and styling", 0.24 if furniture else 0.08),
                ("Lighting and electrical", 0.16),
                ("Labour and installation", 0.16),
                ("Design and coordination", 0.07),
                ("Contingency", 0.05),
            ]
        elif scope == "partial":
            weights = [
                ("Civil, ceiling, and finishes", 0.22),
                ("Modular kitchen or storage", 0.24),
                ("Furniture", 0.18 if furniture else 0.06),
                ("Lighting and electrical", 0.12),
                ("Labour and execution", 0.17),
                ("Design and coordination", 0.07),
                ("Contingency", 0.05),
            ]
        else:
            weights = [
                ("Civil, ceiling, and finishes", 0.24),
                ("Modular kitchen and wardrobes", 0.27),
                ("Furniture", 0.17 if furniture else 0.06),
                ("Lighting and electrical", 0.11),
                ("Labour and execution", 0.15),
                ("Design and coordination", 0.07),
                ("Contingency", 0.05),
            ]

        weight_total = sum(weight for _, weight in weights)
        breakup = []
        assigned = 0
        for label, weight in weights[:-1]:
            amount = round(total * weight / weight_total)
            assigned += amount
            breakup.append((label, amount))
        breakup.append((weights[-1][0], total - assigned))
        return breakup

    @staticmethod
    def _format_breakup(breakup):
        return "; ".join(f"{label} {format_inr(amount)}" for label, amount in breakup)

    def _budget_advice(self, answers, estimate, estimate_fn, budget):
        if not budget:
            return []
        total = int(estimate["total"])
        gap = budget - total
        if gap >= 0:
            cushion = gap / max(total, 1)
            if cushion >= 0.10:
                return [
                    f"your {format_inr(budget)} budget fits with about {format_inr(gap)} cushion."
                ]
            return [
                f"your {format_inr(budget)} budget fits, but the cushion is only {format_inr(gap)}."
            ]

        lines = [
            f"your {format_inr(budget)} budget is short by about {format_inr(abs(gap))}."
        ]
        savings = self._savings_options(answers, estimate, estimate_fn)
        if savings:
            label, saving, _ = savings[0]
            lines.append(f"Biggest model-backed lever: {label}, saving about {format_inr(saving)}.")
        return lines

    def _planning_suggestions(self, answers, estimate, estimate_fn, budget):
        suggestions = []
        savings = self._savings_options(answers, estimate, estimate_fn)

        if budget and estimate["total"] > budget and savings:
            for label, saving, _ in savings[:2]:
                suggestions.append(f"{label} can reduce the estimate by about {format_inr(saving)}.")

        if answers.get("materials_grade") in {"premium", "luxury"}:
            suggestions.append("Use the higher grade in kitchen, wardrobes, and visible surfaces; keep secondary areas one grade lower.")
        if answers.get("scope") == "full":
            suggestions.append("Phase loose furniture and decor after core civil, modular, and electrical work.")
        if answers.get("furniture_included"):
            suggestions.append("Lock modular furniture dimensions early because carpentry changes usually move both cost and timeline.")
        if not suggestions:
            suggestions.append("Keep a 10-12% contingency for brand changes, site fixes, and installation extras.")
        return suggestions[:3]

    def _savings_options(self, answers, estimate, estimate_fn):
        variants = []
        grade = answers.get("materials_grade")
        if grade in GRADE_ORDER and GRADE_ORDER.index(grade) > 0:
            lower_grade = GRADE_ORDER[GRADE_ORDER.index(grade) - 1]
            variants.append((f"Switch to {lower_grade} materials", {"materials_grade": lower_grade}))
        if answers.get("scope") == "full":
            variants.append(("Use partial scope instead of full", {"scope": "partial"}))
        elif answers.get("scope") == "partial":
            variants.append(("Use decor-only scope for non-core rooms", {"scope": "decor only"}))
        if answers.get("furniture_included"):
            variants.append(("Exclude or phase modular furniture", {"furniture_included": 0}))
        if int(answers.get("n_materials", 1)) > 2:
            variants.append(("Reduce one material family", {"n_materials": int(answers["n_materials"]) - 1}))

        base_total = int(estimate["total"])
        options = []
        for label, updates in variants:
            scenario_answers = dict(answers)
            scenario_answers.update(updates)
            try:
                scenario_estimate = estimate_fn(self._prediction_payload(scenario_answers))
            except Exception:
                continue
            saving = base_total - int(scenario_estimate["total"])
            if saving > 0:
                options.append((label, saving, scenario_estimate))
        return sorted(options, key=lambda item: item[1], reverse=True)

    def _project_brief(self, answers):
        city = answers.get("city", "the selected city")
        city_tier = answers.get("city_tier") or CITY_TIER.get(str(city).lower(), "tier2")
        furniture = "with modular furniture" if answers.get("furniture_included") else "without modular furniture"
        return (
            f"{str(answers.get('project_type')).title()} project in {city} "
            f"({city_tier}), {int(answers.get('property_area_sqft')):,} sqft, "
            f"{int(answers.get('rooms'))} rooms/zones, {answers.get('scope')} scope, "
            f"{answers.get('materials_grade')} materials, {furniture}, "
            f"{int(answers.get('estimated_labour_days'))} labour days, "
            f"{answers.get('contractor_type')} execution, {answers.get('design_style')} style, "
            f"{int(answers.get('n_materials'))} material families."
        )
