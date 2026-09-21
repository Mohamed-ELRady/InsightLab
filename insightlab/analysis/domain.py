"""Domain-first interpretation of an uploaded dataset.

The profiler knows *types*; this module adds the missing semantic layer.  Its
rules are intentionally conservative: a numeric column is not revenue merely
because it is the first number in a file, and observational measurements are
never summed unless their meaning explicitly supports it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ..core.state import Clarification, DatasetProfile, DatasetUnderstanding, Role


@dataclass(frozen=True)
class DomainSpec:
    family: str
    domain_en: str
    domain_ar: str
    title_en: str
    title_ar: str
    subject: str
    row_en: str
    row_ar: str
    signals: tuple[str, ...]
    primary: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()


SPECS: tuple[DomainSpec, ...] = (
    DomainSpec(
        "earth_science", "seismology and earth science", "علم الزلازل وعلوم الأرض",
        "Earthquake activity record", "سجل النشاط الزلزالي", "earthquake events",
        "a recorded seismic event", "حدثًا زلزاليًا مسجلًا",
        ("earthquake", "seismic", "quake", "magnitude", "magtype", "depth", "epicenter", "horizontalerror", "deptherror", "magnst"),
        ("mag", "magnitude", "depth"), ("magtype", "type", "status", "place", "net"),
    ),
    DomainSpec(
        "weather", "weather and environmental observations", "الطقس والرصد البيئي",
        "Environmental observations", "سجل الرصد البيئي", "environmental observations",
        "an observation at a place and time", "قراءة رصد في موقع ووقت محددين",
        ("temperature", "humidity", "rainfall", "precipitation", "windspeed", "weather", "station", "airquality", "pm25", "pm10"),
        ("temperature", "rainfall", "precipitation", "humidity", "pm25"), ("station", "weather", "condition", "location"),
    ),
    DomainSpec(
        "health", "health and medical records", "البيانات الصحية والطبية",
        "Health records", "سجل البيانات الصحية", "health records",
        "a patient, visit, test or clinical event", "مريضًا أو زيارة أو فحصًا أو حدثًا سريريًا",
        ("patient", "diagnosis", "treatment", "clinical", "bloodpressure", "heartrate", "symptom", "disease", "hospital"),
        ("heartrate", "bloodpressure", "age", "duration"), ("diagnosis", "treatment", "outcome", "gender", "hospital"),
    ),
    DomainSpec(
        "education", "education and learning", "التعليم والتعلّم",
        "Learning and assessment data", "بيانات التعلّم والتقييم", "education records",
        "a student, assessment or learning activity", "طالبًا أو تقييمًا أو نشاطًا تعليميًا",
        ("student", "grade", "score", "course", "attendance", "school", "teacher", "exam"),
        ("score", "grade", "attendance"), ("course", "class", "school", "subject"),
    ),
    DomainSpec(
        "technology", "systems, devices and telemetry", "الأنظمة والأجهزة والقياسات التقنية",
        "System and telemetry events", "سجل الأنظمة والقياسات التقنية", "system events",
        "a system, device, log or sensor event", "حدث نظام أو جهاز أو سجل أو مستشعر",
        ("device", "sensor", "latency", "cpu", "memoryusage", "errorcode", "loglevel", "requestid", "uptime"),
        ("latency", "cpu", "memoryusage", "duration", "value"), ("device", "status", "loglevel", "service", "region"),
    ),
    DomainSpec(
        "sports", "sports and athletic performance", "الرياضة والأداء البدني",
        "Sports performance record", "سجل الأداء الرياضي", "sports events or performances",
        "a match, player performance or sporting event", "مباراة أو أداء لاعب أو حدثًا رياضيًا",
        ("player", "team", "match", "score", "goals", "assists", "minutesplayed", "opponent", "season", "tournament"),
        ("score", "goals", "assists", "minutesplayed", "distance"), ("player", "team", "opponent", "season", "position"),
    ),
    DomainSpec(
        "transport", "transportation and mobility", "النقل والحركة",
        "Mobility and transport record", "سجل النقل والحركة", "transport journeys or events",
        "a trip, vehicle observation or transport event", "رحلة أو ملاحظة مركبة أو حدث نقل",
        ("trip", "vehicle", "route", "origin", "destination", "speed", "distance", "traffic", "arrival", "departure"),
        ("duration", "distance", "speed", "delay"), ("route", "vehicle", "origin", "destination", "status"),
    ),
    DomainSpec(
        "public_safety", "public safety and incident records", "السلامة العامة وسجلات الحوادث",
        "Public safety incidents", "سجل حوادث السلامة العامة", "incidents",
        "a reported incident or safety event", "بلاغًا أو حادثًا متعلقًا بالسلامة",
        ("incident", "crime", "offense", "casetype", "victim", "suspect", "severity", "police", "emergency"),
        ("severity", "response_time", "duration"), ("incidenttype", "offense", "status", "district", "location"),
    ),
    DomainSpec(
        "agriculture", "agriculture and crop observations", "الزراعة ورصد المحاصيل",
        "Agricultural observations", "سجل الرصد الزراعي", "agricultural observations",
        "a field, crop, harvest or agricultural observation", "حقلًا أو محصولًا أو حصادًا أو ملاحظة زراعية",
        ("crop", "yield", "harvest", "soil", "fertilizer", "irrigation", "farm", "rainfall", "pesticide"),
        ("yield", "rainfall", "moisture", "temperature"), ("crop", "field", "farm", "soiltype", "season"),
    ),
    DomainSpec(
        "energy", "energy generation and consumption", "إنتاج الطاقة واستهلاكها",
        "Energy measurements", "سجل قياسات الطاقة", "energy measurements",
        "a meter reading, generation interval or energy event", "قراءة عداد أو فترة توليد أو حدث طاقة",
        ("energy", "power", "electricity", "voltage", "current", "kilowatt", "kwh", "generation", "consumption", "meter"),
        ("energy", "power", "consumption", "generation", "voltage"), ("meter", "source", "plant", "region", "tariff"),
    ),
    DomainSpec(
        "manufacturing", "manufacturing and quality control", "التصنيع ومراقبة الجودة",
        "Production and quality record", "سجل الإنتاج والجودة", "production records",
        "a produced unit, batch, machine cycle or quality check", "وحدة إنتاج أو دفعة أو دورة آلة أو فحص جودة",
        ("machine", "batch", "defect", "production", "downtime", "cycle_time", "quality", "plant", "workorder"),
        ("cycle_time", "downtime", "defects", "output"), ("machine", "batch", "defecttype", "plant", "shift"),
    ),
    DomainSpec(
        "laboratory", "laboratory and experimental measurements", "القياسات المعملية والتجريبية",
        "Experimental measurements", "سجل القياسات التجريبية", "experimental observations",
        "a sample, experimental run or measurement", "عينة أو تجربة أو قياسًا",
        ("sample", "experiment", "assay", "concentration", "control", "treatment", "replicate", "specimen", "measurement"),
        ("measurement", "concentration", "value", "response"), ("sample", "treatment", "control", "group", "batch"),
    ),
    DomainSpec(
        "population", "population and demographic records", "السكان والسجلات الديموغرافية",
        "Population records", "سجل البيانات السكانية", "population records",
        "a person, household, area or demographic observation", "فردًا أو أسرة أو منطقة أو ملاحظة سكانية",
        ("population", "household", "demographic", "census", "birth", "death", "migration", "occupation", "maritalstatus"),
        ("population", "age", "householdsize"), ("gender", "occupation", "region", "education", "maritalstatus"),
    ),
    DomainSpec(
        "media", "media, content and audience activity", "المحتوى الإعلامي ونشاط الجمهور",
        "Content and audience record", "سجل المحتوى والجمهور", "content or audience events",
        "a piece of content, interaction or audience event", "محتوى أو تفاعلًا أو حدث جمهور",
        ("post", "content", "author", "sentiment", "engagement", "likes", "comments", "shares", "impressions", "hashtag"),
        ("engagement", "likes", "comments", "shares", "impressions"), ("platform", "author", "sentiment", "contenttype", "topic"),
    ),
    DomainSpec(
        "markets", "financial market observations", "رصد الأسواق المالية",
        "Market observations", "سجل رصد الأسواق", "market observations",
        "a security price or market observation at a point in time", "سعر أصل أو ملاحظة سوقية في وقت محدد",
        ("ticker", "symbol", "open", "close", "high", "low", "volume", "marketcap", "return", "volatility"),
        ("close", "price", "return", "volume", "volatility"), ("ticker", "symbol", "exchange", "sector"),
    ),
    DomainSpec(
        "ecology", "ecology and biodiversity observations", "البيئة والتنوع الحيوي",
        "Ecological observations", "سجل الرصد البيئي الحيوي", "ecological observations",
        "a species, habitat, survey or ecological observation", "نوعًا أو موطنًا أو مسحًا أو ملاحظة بيئية",
        ("species", "habitat", "biodiversity", "abundance", "ecosystem", "survey", "vegetation", "wildlife", "genus"),
        ("abundance", "count", "biomass", "diversity"), ("species", "habitat", "site", "genus", "survey"),
    ),
    DomainSpec(
        "business", "business and commercial activity", "النشاط التجاري والأعمال",
        "Business activity", "سجل النشاط التجاري", "business records",
        "a transaction, customer, product or operational record", "معاملة أو عميلًا أو منتجًا أو سجلًا تشغيليًا",
        ("revenue", "sales", "sale", "orderid", "invoice", "customerid", "product", "profit", "price", "quantity", "expense"),
        ("revenue", "sales", "amount", "profit", "price"), ("product", "category", "customersegment", "channel", "region"),
    ),
)


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06ff]+", "", str(value).casefold())


def _find(names: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {name: _norm(name) for name in names}
    for candidate in candidates:
        wanted = _norm(candidate)
        for name, value in normalized.items():
            if value == wanted:
                return name
    for candidate in candidates:
        wanted = _norm(candidate)
        for name, value in normalized.items():
            if wanted and wanted in value:
                return name
    return ""


def _score(spec: DomainSpec, names: list[str], source_name: str) -> tuple[int, list[str]]:
    haystack = [_norm(name) for name in names] + [_norm(source_name)]
    matched = [signal for signal in spec.signals if any(_norm(signal) in value for value in haystack)]
    # Several highly characteristic seismology columns together must beat a
    # stray generic word such as "status" or "type".
    return len(matched), matched


def infer_family_from_text(text: str) -> str:
    """Classify an explicit user description when the schema was ambiguous."""
    normalized = _norm(text)
    ranked = sorted(
        (
            sum(_norm(signal) in normalized for signal in spec.signals),
            spec.family,
        )
        for spec in SPECS
    )
    score, family = ranked[-1]
    return family if score else "general"


def _period(profile: DatasetProfile, date: str) -> tuple[str, str]:
    column = profile.column(date) if date else None
    stats = column.stats if column else {}
    return str(stats.get("earliest", "")), str(stats.get("latest", ""))


def _fmt(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def infer_understanding(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    source_name: str,
    *,
    language_code: str = "en",
) -> DatasetUnderstanding:
    """Return a grounded, domain-aware reading without calling an LLM."""
    names = [str(name) for name in frame.columns]
    ranked = sorted(
        ((_score(spec, names, source_name)[0], spec, _score(spec, names, source_name)[1]) for spec in SPECS),
        key=lambda item: item[0], reverse=True,
    )
    score, spec, matched = ranked[0]
    # Requiring two signals avoids turning an arbitrary table into business
    # data because it happens to contain a column called "value" or "status".
    known = score >= 2
    ar = language_code == "ar"

    measures = profile.names_with_role(Role.MEASURE)
    dates = profile.names_with_role(Role.DATETIME)
    categories = profile.names_with_role(Role.CATEGORY, Role.BOOLEAN)
    if known:
        primary = _find(measures, spec.primary) or (measures[0] if measures else "")
        category = _find(categories, spec.categories) or (categories[0] if categories else "")
        domain = spec.domain_ar if ar else spec.domain_en
        title = spec.title_ar if ar else spec.title_en
        subject = spec.subject
        row = spec.row_ar if ar else spec.row_en
        family = spec.family
        confidence = "high" if score >= 4 else "medium"
        if spec.family == "business" and any(
            token in {_norm(item) for item in matched}
            for token in ("revenue", "sales", "sale", "orderid", "invoice")
        ):
            domain = "معاملات المبيعات" if ar else "sales transactions"
            title = "تحليل معاملات المبيعات" if ar else "Sales transaction analysis"
            subject = "sales transactions"
            row = "عملية بيع أو طلب أو بند فاتورة" if ar else "a sale, order or invoice line"
    else:
        primary = measures[0] if measures else ""
        category = categories[0] if categories else ""
        domain = "بيانات عامة متعددة المجالات" if ar else "general multi-domain data"
        title = "استكشاف مجموعة البيانات" if ar else "Dataset discovery"
        subject = "records or observations"
        row = "سجلًا أو ملاحظة واحدة" if ar else "one record or observation"
        family = "general"
        confidence = "low"

    primary_date = dates[0] if dates else ""
    aggregations: dict[str, str] = {}
    for name in measures:
        lowered = _norm(name)
        additive = family == "business" and any(
            token in lowered for token in ("revenue", "sales", "amount", "profit", "cost", "expense", "quantity", "units")
        )
        aggregations[name] = "sum" if additive else "mean"

    earliest, latest = _period(profile, primary_date)
    period_text = ""
    if earliest and latest:
        period_text = (
            f" وتمتد من {earliest} إلى {latest}" if ar
            else f", spanning {earliest} to {latest}"
        )

    if ar:
        summary = (
            f"الملف يتناول {domain}. يحتوي على {len(frame):,} {row} عبر "
            f"{len(names)} عمودًا{period_text}. "
            + (f"المقياس الأهم مبدئيًا هو {primary}، وسيُحلل كقياس رصدي بالمتوسط والتوزيع المناسبين. " if primary else "")
            + "ستُبنى الأسئلة والرسوم والمؤشرات التالية على هذا المعنى."
        )
    else:
        summary = (
            f"This is {domain} data with {len(frame):,} rows across {len(names)} columns{period_text}. "
            + (f"The leading measure is {primary}; it will be analysed as an observation with appropriate averages and distributions. " if primary and family != "business" else "")
            + "The next questions, charts and headline figures will follow this meaning."
        )

    hook = ""
    if primary:
        values = pd.to_numeric(frame[primary], errors="coerce").dropna()
        if not values.empty:
            maximum = float(values.max())
            median = float(values.median())
            if family == "earth_science" and _norm(primary) in {"mag", "magnitude"}:
                place = ""
                place_column = _find(names, ("place", "location"))
                if place_column:
                    idx = pd.to_numeric(frame[primary], errors="coerce").idxmax()
                    place = str(frame.loc[idx, place_column])
                hook = (
                    f"أقوى حدث مسجل بلغ {_fmt(maximum)} على مقياس الزلزال"
                    + (f" قرب {place}" if place else "") + " — وهذه نقطة البداية لفهم نمط النشاط كله."
                    if ar else
                    f"The strongest recorded event reached {_fmt(maximum)}"
                    + (f" near {place}" if place else "") + "—the starting point for understanding the full activity pattern."
                )
            else:
                hook = (
                    f"أول إشارة لافتة: يتراوح {primary} حول قيمة وسيطة {_fmt(median)} ويصل إلى {_fmt(maximum)}."
                    if ar else f"First signal: {primary} has a median of {_fmt(median)} and reaches {_fmt(maximum)}."
                )
    if not hook:
        hook = (
            f"أمامك {len(frame):,} سجلًا؛ سنحوّلها الآن من صفوف خام إلى قصة قابلة للفهم والتحقق."
            if ar else f"There are {len(frame):,} records here; the next step turns them from raw rows into a testable story."
        )

    if family == "earth_science":
        questions = ([
            "أين ومتى تركزت الأحداث الأقوى؟",
            "هل تغيّر عدد الزلازل أو شدتها بمرور الوقت؟",
            "ما العلاقة بين العمق والقوة؟",
        ] if ar else [
            "Where and when did the strongest events concentrate?",
            "Did earthquake frequency or magnitude change over time?",
            "How are depth and magnitude related?",
        ])
        goal = "فهم تواتر الزلازل وقوتها وعمقها وتوزيعها المكاني" if ar else "Understand earthquake frequency, magnitude, depth and spatial distribution"
    else:
        questions = ([
            f"كيف يتغير {primary} بمرور الوقت؟" if primary_date and primary else "ما الأنماط الأوضح في البيانات؟",
            f"كيف يختلف {primary} بين مجموعات {category}؟" if primary and category else "ما أهم الفروق بين المجموعات؟",
            "هل توجد علاقات أو حالات غير معتادة تستحق التحقق؟",
        ] if ar else [
            f"How does {primary} change over time?" if primary_date and primary else "What are the clearest patterns in the data?",
            f"How does {primary} differ across {category}?" if primary and category else "Which groups differ most?",
            "Are there relationships or unusual cases worth checking?",
        ])
        goal = "اكتشاف الأنماط والفروق والعلاقات المهمة" if ar else "Discover the important patterns, differences and relationships"

    clarifications: list[Clarification] = []
    if not known:
        clarifications.append(Clarification(
            kind="row_meaning",
            question="ما موضوع هذه البيانات، وما الذي يمثله كل صف؟" if ar else "What is this data about, and what does one row represent?",
            reason="الإجابة تحدد القياسات والتجميعات والأسئلة الصحيحة بدل افتراض مجال غير موجود." if ar else "This determines the right measures, aggregations and questions instead of assuming a domain.",
        ))

    important = [name for name in (primary, primary_date, category) if name]
    important.extend(name for name in measures if name not in important)
    return DatasetUnderstanding(
        business_domain=domain,
        domain_family=family,
        dataset_title=title,
        subject=subject,
        row_represents=row,
        analysis_goal=goal,
        hook=hook,
        confidence=confidence,
        summary=summary,
        important_columns=important[:6],
        primary_measure=primary,
        primary_date=primary_date,
        primary_category=category,
        measure_aggregations=aggregations,
        suggested_questions=questions,
        questions=clarifications,
        source="rules",
    )
