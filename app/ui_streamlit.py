
"""Sitr | سِتر privacy workspace.

    streamlit run app/ui_streamlit.py

Four focused pages share one privacy-first dashboard:

  1. Upload  - pick a document, choose the receiver, say why it is being shared.
               The model panel shows whether the AI is installed and on what hardware,
               because in this version there is no analysis without it.
  2. Review  - original and safe version side by side, one card per item, and on every
               card: the rule the model cited, the model's own confidence, and - when the
               model was not trusted - the exact reason it was sent to you instead.
  3. Audit   - every decision with its rule and reason, the run's AI summary, and the
               model's footprint: which jobs it was asked to do, how long each took, how
               many tokens came back.

Two behaviours are deliberate and worth knowing when you read this file:

* if the model is not installed, the Upload page does not offer an analysis at all - it
  tells you the exact command to install it. There is no rules-only path to fall back to,
  and pretending otherwise would misrepresent what the system is;
* the model is only called when you press Analyze. Opening the dashboard never triggers
  document analysis.
"""
from __future__ import annotations

import html
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config                                      # noqa: E402
from app.audit import store as audit                        # noqa: E402
from app.core import analyze, apply_decisions, write_summary  # noqa: E402
from app.document_processing import readers                 # noqa: E402
from app.document_processing.writers import safe_text, write_release  # noqa: E402
from app.llm import engine                                  # noqa: E402
from app.pii_detection.schema import Action, PiiType        # noqa: E402
from app.policy import qa as policy_qa                      # noqa: E402
from app.policy import document_qa                         # noqa: E402
from app.policy.policy_editor import PolicyEditError, PolicyEditor  # noqa: E402
from app.policy.policy_index import PolicyIndexError, get_index  # noqa: E402
from app.review.review_queue import apply_review, build_queue, review_state  # noqa: E402
from app.rules.store import library as default_library      # noqa: E402

SAMPLE_DIR = config.DATA_DIR / "synthetic"
_ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="safeshare-analysis")
PRODUCT_BRAND = "Sitr | سِتر"

ACTION_STYLE = {
    "KEEP": ("#0f5132", "#d1e7dd"),
    "REPLACE": ("#664d03", "#fff3cd"),
    "REMOVE": ("#842029", "#f8d7da"),
    "ASK_HUMAN": ("#78581b", "#f8e7b1"),
}
PENDING_STYLE = ACTION_STYLE["ASK_HUMAN"]
CHART_COLORS = {
    "KEEP": "#40d6ae", "REPLACE": "#ffb454", "REMOVE": "#ff647c",
    "ASK_HUMAN": "#e1bc68", "OTHER": "#7290ad",
}

ARABIC_TRANSLATIONS = {
    'Privacy workspace': 'مساحة سِتر للخصوصية',
    'AI governance · secure release': 'خصوصية البيانات · مشاركة آمنة',
    'Upload': 'رفع الملف',
    'Review': 'المراجعة',
    'Audit log': 'سجل التدقيق',
    'Policy': 'السياسات',
    'Workspace': 'مساحة العمل',
    'Language': 'اللغة',
    'SAFE SHARE': 'مشاركة آمنة',
    'PRIVACY INTELLIGENCE LAB': 'مختبر خصوصية البيانات',
    'A perimeter': 'حماية تحيط',
    'for every release.': 'بكل مشاركة.',
    'Inspect sensitive data. Apply recipient-specific rules. Approve a release with evidence attached.': 'اكتشف البيانات الحساسة، وطبّق قواعد الجهة المستلمة، وراجع نسخة مشاركة موثقة.',
    'Privacy shield, ready': 'درع الخصوصية جاهز',
    'Private data': 'بيانات خاصة',
    'Protected release': 'نسخة محمية',
    'RELEASE SEQUENCE': 'مراحل المشاركة',
    'AWAITING SOURCE DOCUMENT': 'بانتظار المستند',
    'ANALYSIS IN PROGRESS': 'جارٍ الفحص',
    'REVIEW READY': 'جاهز للمراجعة',
    'Detect PII': 'اكتشاف البيانات الحساسة',
    'Apply policy': 'تطبيق السياسة',
    'AI decisions': 'قرارات المعالجة',
    'Review decisions': 'مراجعة القرارات',
    'Safe release': 'النسخة الآمنة',
    '01 / Choose the recipient': '01 / اختر الجهة المستلمة',
    '02 / Add a document': '02 / أضف مستنداً',
    'SHARING PURPOSE': 'الغرض من المشاركة',
    'Internal HR': 'الموارد البشرية',
    'External Vendor': 'مورد خارجي',
    'Analyst': 'محلل',
    'Customer Support': 'دعم العملاء',
    'Custom receiver': 'جهة مخصصة',
    'POLICY PROFILE': 'ملف السياسة',
    'live rules': 'قواعد فعالة',
    'No rules defined': 'لا توجد قواعد محددة',
    'KEEP': 'إبقاء',
    'REPLACE': 'استبدال',
    'REMOVE': 'حذف',
    'ASK_HUMAN': 'مراجعة بشرية',
    'REVIEW': 'مراجعة',
    'Selected': 'محدد',
    'Choose': 'اختيار',
    'Drop a document or browse': 'أفلت المستند أو استعرض الملفات',
    'PDF, DOCX and TXT are supported.': 'الصيغ المدعومة: PDF وDOCX وTXT.',
    'Or use a synthetic sample': 'أو استخدم مستنداً تجريبياً',
    'none': 'بدون اختيار',
    'Synthetic sample selected · no external file is required.': 'تم اختيار مستند تجريبي · لا حاجة إلى ملف خارجي.',
    'Analyze document  →': 'افحص المستند  ←',
    'Document text may be processed by an external service.': 'قد يُعالج نص المستند عبر خدمة خارجية.',
    'Document content stays on this device.': 'يبقى محتوى المستند على هذا الجهاز.',
    'Analysis is unavailable. Ask your administrator to configure the privacy service.': 'الفحص غير متاح حالياً. تواصل مع مسؤول النظام لإعداد الخدمة.',
    'This file type is not supported.': 'نوع هذا الملف غير مدعوم.',
    'Upload a document or choose a sample first.': 'ارفع مستنداً أو اختر نموذجاً أولاً.',
    'Checking this document against recipient safeguards…': 'جارٍ فحص المستند وفق ضوابط الجهة المستلمة…',
    'Preparing document review.': 'جارٍ تجهيز مراجعة المستند.',
    'The review could not be completed. No document was changed.': 'تعذرت إكمال المراجعة. لم يتم تعديل المستند.',
    'Privacy review': 'مراجعة الخصوصية',
    'Nothing to review yet.': 'لا توجد عناصر للمراجعة بعد.',
    'Run an analysis to compare the original with a receiver-safe release.': 'افحص مستنداً لمقارنة الأصل بالنسخة المناسبة للجهة المستلمة.',
    'Go to upload': 'الانتقال إلى رفع المستند',
    'Detected': 'تم اكتشافها',
    'Replaced': 'تم استبدالها',
    'Removed': 'تم حذفها',
    'Retained': 'تم الإبقاء عليها',
    'Human review': 'مراجعة بشرية',
    'Protected fields': 'بيانات محمية',
    'Privacy exposure comparison': 'مقارنة حماية البيانات',
    'Items protected': 'العناصر المحمية',
    'Original': 'الأصل',
    'Original · click to reveal': 'الأصل · اضغط للكشف',
    'Safe release preview': 'معاينة النسخة الآمنة',
    'Sensitive source spans are visually redacted. Select a redacted span to inspect its original value.': 'تُحجب البيانات الحساسة في الأصل. اضغط على حقل محجوب لمعاينة قيمته.',
    'Original vs. safe release': 'الأصل والنسخة الآمنة',
    'Decision center': 'مركز القرارات',
    'All types': 'كل الأنواع',
    'All actions': 'كل الإجراءات',
    'Minimum confidence': 'حد الثقة الأدنى',
    'PII type': 'نوع البيانات',
    'Action': 'الإجراء',
    'Confidence': 'الثقة',
    'No findings match these filters.': 'لا تطابق النتائج هذه المرشحات.',
    'Every item was decided from the policy. Nothing needs a person.': 'تم حسم جميع العناصر وفق السياسة. لا توجد عناصر بانتظار المراجعة البشرية.',
    'Waiting for your decision': 'بانتظار قرارك',
    'No open review items match these filters.': 'لا توجد عناصر مفتوحة تطابق هذه المرشحات.',
    'Keep': 'إبقاء',
    'Replace': 'استبدال',
    'Remove': 'حذف',
    'Reject suggestion': 'رفض الاقتراح',
    'Model was unsure': 'يتطلب مراجعة',
    "Accept the model's proposal for the remaining items": 'اعتماد المقترح لبقية العناصر',
    'Generate release': 'إنشاء النسخة الآمنة',
    'Download the safe version': 'تنزيل النسخة الآمنة',
    'Safe version as text': 'النسخة الآمنة كنص',
    'Your decision': 'قرارك',
    'Reason': 'السبب',
    'No valid rule': 'لا توجد قاعدة مطابقة',
    'Policy rule': 'قاعدة السياسة',
    'Validated synthetic substitute': 'بديل تجريبي تم التحقق منه',
    'No audit events yet. Generate a release from the Review page to populate analytics.': 'لا توجد أحداث مسجلة بعد. أنشئ نسخة آمنة لتظهر بيانات النشاط هنا.',
    'Audit intelligence': 'سجل الخصوصية',
    'Release activity and recorded decisions.': 'نشاط المشاركة والقرارات المسجلة.',
    'Receiver': 'الجهة المستلمة',
    'All receivers': 'كل الجهات',
    'Document': 'المستند',
    'All documents': 'كل المستندات',
    'Date range': 'النطاق الزمني',
    'Runs': 'عمليات المشاركة',
    'PII decisions': 'قرارات البيانات',
    'Replacements': 'استبدالات',
    'Removals': 'عمليات حذف',
    'Review requests': 'طلبات مراجعة',
    'Decision timeline': 'الخط الزمني للقرارات',
    'Document not recorded': 'مستند غير مسجل',
    'Receiver not recorded': 'جهة غير مسجلة',
    'Run id not recorded': 'معرّف غير مسجل',
    'Timestamp not recorded': 'وقت غير مسجل',
    'No recorded decisions match these filters.': 'لا توجد قرارات مسجلة تطابق هذه المرشحات.',
    'Decision distribution': 'توزيع القرارات',
    'Protected data categories': 'فئات البيانات المحمية',
    'Activity timeline': 'الخط الزمني للنشاط',
    'Activity by receiver': 'النشاط حسب الجهة',
    'No matching run-level activity.': 'لا يوجد نشاط مطابق.',
    'Release activity': 'نشاط المشاركة',
    'Time': 'الوقت',
    'Masked value': 'قيمة محجوبة',
    'Decision source': 'مصدر القرار',
    'Download filtered, masked audit CSV': 'تنزيل سجل التدقيق المحجوب',
    'Policy and decision trace': 'سجل السياسة والقرارات',
    'Policy intelligence': 'استكشاف السياسات',
    'One policy. Every release.': 'سياسة واحدة. لكل مشاركة.',
    'Explore receiver coverage, ask grounded questions, and audit policy changes.': 'استعرض القواعد، واسأل عن السياسة، وتابع التغييرات المسجلة.',
    'Active rules': 'القواعد الفعالة',
    'Receivers': 'الجهات المستلمة',
    'Indexed rules': 'القواعد المتاحة للبحث',
    'Coverage': 'التغطية',
    'Policy search is currently unavailable. The rules below remain editable and releases are unaffected.': 'البحث في السياسات غير متاح حالياً. يمكنك تعديل القواعد أدناه، ولن تتأثر المشاركة.',
    'Ask the policy': 'اسأل عن السياسة',
    'Browse and edit': 'استعراض وتعديل',
    'Change history': 'سجل التغييرات',
    'Released document': 'المستند المُعتمد',
    'Your question': 'سؤالك',
    'May an external vendor see a phone number?': 'هل يمكن لمورد خارجي الاطلاع على رقم هاتف؟',
    '...or pick one': '...أو اختر سؤالاً',
    'any': 'الكل',
    'any receiver': 'كل الجهات',
    'Receiver filter': 'تصفية حسب الجهة',
    'Type filter': 'تصفية حسب النوع',
    'Write a concise answer': 'صياغة إجابة موجزة',
    'Ask': 'اسأل',
    'Searching the policy...': 'جارٍ البحث في السياسة...',
    'Answer': 'الإجابة',
    'Verified citations: ': 'مراجع تم التحقق منها: ',
    'Filters used:': 'المرشحات المستخدمة:',
    'Source': 'المصدر',
    'Relevance': 'مدى الصلة',
    'Send this to the Review page': 'إرسال إلى صفحة المراجعة',
    'A policy answer is context for a reviewer, never an instruction to the release pipeline. Nothing here changes a decision or a document.': 'إجابة السياسة مرجع للمراجع فقط، ولا تغيّر أي قرار أو مستند.',
    'all': 'الكل',
    'all receivers': 'كل الجهات',
    'Change a rule': 'تعديل قاعدة',
    'This receiver has no rules. Every sensitive item it receives comes back as ASK_HUMAN - add a rule below if that is not what you want.': 'لا توجد قواعد لهذه الجهة. ستُحال كل معلومة حساسة إلى المراجعة البشرية؛ أضف قاعدة عند الحاجة.',
    'Rule': 'القاعدة',
    'Rule id (fixed)': 'معرّف القاعدة (ثابت)',
    'Type (fixed)': 'النوع (ثابت)',
    'Receiver (fixed)': 'الجهة (ثابت)',
    'Policy source': 'مرجع السياسة',
    'Your name (recorded in the change history)': 'اسمك (يُسجل في سجل التغييرات)',
    'Save the new policy': 'حفظ السياسة',
    'Delete this rule': 'حذف هذه القاعدة',
    'Delete the rule': 'حذف القاعدة',
    'Add a rule': 'إضافة قاعدة',
    'New rule id': 'معرّف القاعدة الجديدة',
    'Add the rule': 'إضافة القاعدة',
    'Change records': 'سجلات التغيير',
    'Inspect one change': 'عرض تغيير',
    'Before': 'قبل',
    'After': 'بعد',
    'Generate a release on the Review page first. This tab then summarises it or answers questions about it.': 'أنشئ النسخة الآمنة من صفحة المراجعة أولاً، ثم يمكنك تلخيصها أو السؤال عنها.',
    'Released document assistant': 'مساعد المستند المُعتمد',
    'Ask about the document': 'اسأل عن المستند',
    'Summarize': 'تلخيص',
    'Clear conversation': 'مسح المحادثة',
    'Summary': 'الملخص',
    'Who signed the agreement? / من الطرف الثاني في العقد؟': 'من وقّع الاتفاقية؟',
    'The released text may be processed by an external service.': 'قد يُعالج نص النسخة المُعتمدة عبر خدمة خارجية.',
    'The rules that match this question': 'القواعد المطابقة لهذا السؤال',
    'Recipient': 'الجهة المستلمة',
    'Supported formats': 'الصيغ المدعومة',
    'Privacy controls': 'ضوابط الخصوصية',
    'active safeguards': 'ضوابط فعالة',
    'Review and audit': 'مراجعة وتدقيق',
    'Analysis complete': 'اكتمل الفحص',
    'findings checked': 'عناصر تم فحصها',
    'chunks complete': 'أجزاء مكتملة',
    'elapsed': 'الوقت المنقضي',
    'No source citation': 'لا يوجد مرجع للمصدر',
    'The matching rules are shown below.': 'القواعد المطابقة ظاهرة أدناه.',
    'SAFE SHARE WORKSPACE': 'مساحة المشاركة الآمنة',
    'PRIVACY REVIEW': 'مراجعة الخصوصية',
    'REVIEW WORKSPACE': 'مساحة المراجعة',
    'SECURITY ANALYTICS': 'تحليلات الخصوصية',
    'Evidence from recorded release runs, policy decisions and actual model calls.': 'نشاط المشاركة والقرارات المسجلة.',
    'Guardrail activity': 'ملاحظات الحماية',
    'Coral marks indicate findings. On the right, amber is a substitute, green is retained by policy, and a red marker shows a removed span. Left: what the model found, marked on the original text. Right: the released text itself — a highlighted value is what goes into the document (yellow = a substitute, green = kept), and a thin mark shows where a value was removed, because in the release it is gone. Nothing still waiting for you is in the release.': 'تشير العلامات إلى البيانات التي عثر عليها الفحص. تعرض الجهة اليمنى النسخة الناتجة: البدائل مميزة، وما حُذف لا يظهر فيها. العناصر التي تنتظر قراراً بشرياً لا تُدرج في النسخة.',
    'Model and policy decisions': 'قرارات السياسة',
    'Policy · ': 'السياسة · ',
    'Rule the model cited:': 'القاعدة المرجعية:',
    "Model's reason:": 'السبب:',
    'Model confidence': 'مستوى الثقة',
    'Asked because': 'سبب الإحالة:',
    'Apply nothing the model proposed. The item stays open and out of the release.': 'لن يُطبّق أي اقتراح. سيبقى العنصر مفتوحاً ومستبعداً من النسخة.',
    'Faster for a demo. Each answer is still recorded as a human decision in the audit log.': 'يسجل كل اختيار بوصفه قراراً بشرياً في سجل التدقيق.',
    'Format': 'الصيغة',
    'layout preserved': 'تم الحفاظ على التنسيق',
    'Decision-level audit trail': 'سجل قرارات المشاركة',
    'Values are masked. The table omits raw values, model explanations, and replacement text.': 'القيم محجوبة. لا يتضمن الجدول القيم الأصلية أو التفسيرات أو البدائل.',
    'No audit events yet.': 'لا توجد أحداث تدقيق بعد.',
    'Document filter': 'تصفية حسب المستند',
    'Action filter': 'تصفية حسب الإجراء',
    'No recorded run timestamps are available for this selection.': 'لا تتوفر أوقات مسجلة لهذا الاختيار.',
    'Ask in Arabic or English. The answer comes from the rules that match the question, and every statement points at a rule id - the retrieved rules are always shown, so you can check the answer against the policy itself.': 'اسأل بالعربية أو الإنجليزية. تستند الإجابة إلى القواعد المطابقة، وتظهر القواعد المرجعية للتحقق منها.',
    'Source · ': 'المصدر · ',
    'I understand this changes the policy for every future analysis, not just the one on screen.': 'أفهم أن هذا التغيير سيطبق على عمليات الفحص القادمة.',
    'I understand this adds a rule to the live policy.': 'أفهم أن هذه القاعدة ستضاف إلى السياسة الحالية.',
    'Search policy': 'البحث في السياسة',
    'Privacy service unavailable.': 'خدمة الخصوصية غير متاحة حالياً.',
    'Sitr': 'سِتر',
    'Add document': 'إضافة مستند',
    'Scan': 'الفحص',
    'Review findings': 'مراجعة النتائج',
    'Protect document': 'حماية المستند',
    'Release': 'المشاركة',
    'Choose a recipient': 'اختر الجهة المستلمة',
    'Add a document': 'أضف مستنداً',
    'Scan in progress': 'جارٍ فحص المستند',
    'Protect and release': 'حماية ومشاركة',
    'Ready to release': 'جاهز للمشاركة',
    'Share as-is': 'مشاركة كما هي',
    'Use a safe substitute': 'استخدام بديل آمن',
    'Remove from shared copy': 'إزالة من النسخة المشتركة',
    'Needs your review': 'تحتاج إلى مراجعتك',
    'No sharing rules yet': 'لا توجد قواعد مشاركة بعد',
    'Sensitive items will wait for your review.': 'ستبقى البيانات الحساسة بانتظار مراجعتك.',
    'Recipient profile': 'ملف الجهة',
    'Sharing rules': 'قواعد المشاركة',
    'Intended purpose': 'الغرض من المشاركة',
    'Document type': 'نوع المستند',
    'Choose a file': 'اختر ملفاً',
    'Drop your document here or browse': 'أفلت المستند هنا أو استعرض الملفات',
    'Supported formats: PDF, DOCX, TXT': 'الصيغ المدعومة: PDF وDOCX وTXT',
    'Use a sample document': 'استخدم مستنداً تجريبياً',
    'Sample document': 'مستند تجريبي',
    'Start privacy scan': 'بدء فحص الخصوصية',
    'Scan this document': 'افحص هذا المستند',
    'Preparing the privacy scan…': 'جارٍ تجهيز فحص الخصوصية…',
    "Scanning the document against the selected recipient's rules…": 'جارٍ فحص المستند وفق قواعد الجهة المختارة…',
    'The scan is taking longer than expected. You can keep this page open.': 'يستغرق الفحص وقتاً أطول من المتوقع. يمكنك إبقاء الصفحة مفتوحة.',
    'The scan could not be completed. No document was changed.': 'تعذر إكمال الفحص. لم يتم تعديل المستند.',
    'The selected file could not be read.': 'تعذرت قراءة الملف المحدد.',
    'Choose a document before starting the scan.': 'اختر مستنداً قبل بدء الفحص.',
    'Scan complete': 'اكتمل الفحص',
    'items found': 'عناصر مكتشفة',
    'Before protection': 'قبل الحماية',
    'Protected preview': 'معاينة النسخة المحمية',
    'Click a redacted value to reveal it temporarily.': 'اضغط على القيمة المحجوبة لإظهارها مؤقتاً.',
    'The shared copy contains only the decisions shown here.': 'تحتوي النسخة المشتركة على القرارات المعروضة هنا فقط.',
    'Generate protected copy': 'إنشاء النسخة المحمية',
    'Download protected copy': 'تنزيل النسخة المحمية',
    'Your protected copy is ready.': 'نسختك المحمية جاهزة.',
    'Some items still need your decision and are excluded from the shared copy.': 'تحتاج بعض العناصر إلى قرارك، وقد استُبعدت من النسخة المشتركة.',
    'All detected items have a policy-backed decision.': 'لجميع العناصر المكتشفة قرارات مستندة إلى السياسة.',
    'Select a recipient to see its sharing safeguards.': 'اختر جهة مستلمة لمعاينة ضوابط المشاركة الخاصة بها.',
    'All data types': 'كل أنواع البيانات',
    'All decision types': 'كل القرارات',
    'Show only items needing review': 'إظهار العناصر التي تحتاج إلى مراجعة فقط',
    'No findings match your filters.': 'لا توجد نتائج تطابق عوامل التصفية.',
    'No items need a human decision.': 'لا توجد عناصر تحتاج إلى قرار بشري.',
    'Decision saved.': 'تم حفظ القرار.',
    'Reason for review': 'سبب المراجعة',
    'Cited policy rule': 'قاعدة السياسة المرجعية',
    'Validated safe substitute': 'بديل آمن تم التحقق منه',
    'Select an action': 'اختر إجراءً',
    'Audit trail': 'سجل التدقيق',
    'Recent activity': 'النشاط الأخير',
    'Share date': 'تاريخ المشاركة',
    'Shared document': 'المستند المشترك',
    'All recipients': 'كل الجهات المستلمة',
    'All decisions': 'كل القرارات',
    'Filter by date': 'تصفية حسب التاريخ',
    'Protected items': 'عناصر محمية',
    'Human reviews': 'مراجعات بشرية',
    'No recorded activity yet.': 'لا يوجد نشاط مسجل بعد.',
    'No policy changes have been recorded.': 'لم تُسجل أي تغييرات على السياسة.',
    'Search policies': 'البحث في السياسات',
    'Browse policies': 'استعراض السياسات',
    'Policy history': 'سجل السياسة',
    'Document assistant': 'مساعد المستند',
    'Search the policy': 'ابحث في السياسة',
    'Write an answer from the retrieved rules': 'صياغة إجابة من القواعد المسترجعة',
    'Search results': 'نتائج البحث',
    'Rules that support this answer': 'القواعد التي تدعم هذه الإجابة',
    'Policy answer': 'إجابة السياسة',
    'Ask about this released document': 'اسأل عن هذا المستند المُعتمد',
    'Summarize released document': 'تلخيص المستند المُعتمد',
    'Clear messages': 'مسح الرسائل',
    'The assistant only sees the protected copy, never the original.': 'يطلع المساعد على النسخة المحمية فقط، ولا يطلع على الأصل.',
    'The document text may be processed by an external service.': 'قد تتم معالجة نص المستند عبر خدمة خارجية.',
    'The document text stays on this device.': 'يبقى نص المستند على هذا الجهاز.',
    'Connection details': 'تفاصيل الاتصال',
    'Model details': 'تفاصيل المعالجة',
    'Policy details': 'تفاصيل السياسة',
    'Technical details': 'تفاصيل تقنية',
    'Policy version': 'إصدار السياسة',
    'Privacy service': 'خدمة الخصوصية',
    'Processing location': 'موقع المعالجة',
    'API endpoint': 'نقطة اتصال الخدمة',
    'Model name': 'اسم النموذج',
    'Inference time': 'زمن المعالجة',
    'Calls recorded': 'الاستدعاءات المسجلة',
    'Current policy': 'السياسة الحالية',
    'No rules for this recipient': 'لا توجد قواعد لهذه الجهة',
    'The policy does not permit sharing by default.': 'لا تسمح السياسة بالمشاركة تلقائياً.',
    'Keep as-is': 'الإبقاء كما هو',
    'Replace with safe value': 'استبدال بقيمة آمنة',
    'Remove from copy': 'الإزالة من النسخة',
    'Ask a reviewer': 'إحالة إلى المراجع',
    'Confirm policy change': 'تأكيد تغيير السياسة',
    'Confirm rule addition': 'تأكيد إضافة القاعدة',
    'Rule saved.': 'تم حفظ القاعدة.',
    'Rule deleted.': 'تم حذف القاعدة.',
    'A policy update is ready.': 'تم تحديث السياسة.',
    'Change was not saved.': 'لم يتم حفظ التغيير.',
    'English': 'الإنجليزية',
    'Arabic': 'العربية',
    'EN': 'EN',
    'عربي': 'عربي',
    '{count} item(s) need your review. They stay out of the shared copy until decided ({answered}/{total} reviewed).': 'هناك {count} عناصر تحتاج إلى مراجعتك. ستبقى خارج النسخة المشتركة حتى تُحسم ({answered}/{total}).',
    'Count of items protected by recipient policy.': 'عدد العناصر المحمية وفق سياسة الجهة المستلمة.',
    'of detected items protected': 'من العناصر المكتشفة محمية',
    'Every detected item has a policy-backed decision.': 'لجميع العناصر المكتشفة قرار مستند إلى السياسة.',
    'Original document': 'المستند الأصلي',
    'Protected copy preview': 'معاينة النسخة المحمية',
    'Values awaiting your decision are excluded from the shared copy.': 'تُستبعد من النسخة المشتركة القيم التي تنتظر قرارك.',
    'Decision details': 'تفاصيل القرار',
    'Apply nothing from this suggestion. It remains excluded until you decide.': 'لن يُطبّق هذا الاقتراح. سيبقى العنصر مستبعداً حتى تتخذ قرارك.',
    'Accept suggested decisions for remaining items': 'اعتماد القرارات المقترحة لبقية العناصر',
    'Each choice is recorded as a reviewer decision.': 'يُسجل كل اختيار بوصفه قراراً للمراجع.',
    'Document format': 'تنسيق المستند',
    'Layout preserved': 'تم الحفاظ على التنسيق',
    'Changes applied': 'التغييرات المطبقة',
    'Release reference': 'مرجع المشاركة',
    'Open protected text': 'عرض النص المحمي',
    'A protected copy has been created.': 'تم إنشاء نسخة محمية.',
    'Open the review queue': 'فتح قائمة المراجعة',
    'Share safely': 'مشاركة آمنة',
    'Findings': 'النتائج',
    'Why flagged': 'سبب التنبيه',
    'Safeguard': 'إجراء الحماية',
    'Source policy': 'سياسة المصدر',
    'Recipient policy': 'سياسة الجهة المستلمة',
    'This recipient has no sharing rules. Sensitive items are not automatically approved.': 'لا توجد قواعد مشاركة لهذه الجهة. لا تتم الموافقة تلقائياً على البيانات الحساسة.',
    'Private data. Purposeful sharing.': 'بيانات خاصة. مشاركة هادفة.',
    'Safeguard notes': 'ملاحظات الحماية',
    'items need your review': 'عناصر تحتاج إلى مراجعتك',
    'Layout may differ': 'قد يختلف التنسيق',
    'changes applied': 'تغييرات مطبقة',
    'Release note': 'ملاحظة المشاركة',
    'Recorded sharing activity': 'نشاط المشاركة المسجل',
    'Any recipient': 'أي جهة مستلمة',
    'Any type': 'أي نوع',
    'Recipient filter': 'تصفية حسب الجهة',
    'sharing rules': 'قواعد مشاركة',
    'Edit the sharing action, explanation, or source.': 'عدّل إجراء المشاركة أو تفسيره أو مصدره.',
    'Actions in this selection': 'الإجراءات في هذا الاختيار',
    'Sharing action': 'إجراء المشاركة',
    'Sharing explanation': 'تفسير المشاركة',
    'Recipient (fixed)': 'الجهة المستلمة (ثابتة)',
    'The policy change was not saved.': 'لم يتم حفظ تغيير السياسة.',
    'Nothing to update; this rule already has those settings.': 'لا حاجة للتحديث؛ تتضمن القاعدة هذه الإعدادات بالفعل.',
    'The rule was saved, but its audit record could not be written.': 'حُفظت القاعدة، لكن تعذر تسجيلها في سجل التدقيق.',
    'The rule was saved. Policy search will refresh when available.': 'حُفظت القاعدة. سيتجدد البحث في السياسة عند توفره.',
    'The next document scan will use this rule.': 'سيستخدم فحص المستند القادم هذه القاعدة.',
    'Removing this rule means sensitive items will wait for review.': 'عند حذف القاعدة، تنتظر البيانات الحساسة المراجعة.',
    'Type to confirm': 'اكتب للتأكيد',
    'The rule could not be deleted.': 'تعذر حذف القاعدة.',
    'Add a sharing rule': 'إضافة قاعدة مشاركة',
    'Each recipient can have one rule for each data type.': 'يمكن لكل جهة قاعدة واحدة لكل نوع من البيانات.',
    'The new rule could not be saved.': 'تعذر حفظ القاعدة الجديدة.',
    'Policy changes appear newest first with the editor and before/after values.': 'تظهر تغييرات السياسة من الأحدث، مع المحرر والقيم قبل التغيير وبعده.',
    'Change type': 'نوع التغيير',
    'Changes': 'التغييرات',
    'Editor': 'المحرر',
    'Create a protected copy first to use the document assistant.': 'أنشئ نسخة محمية أولاً لاستخدام مساعد المستند.',
    'Protected copy': 'النسخة المحمية',
    'characters': 'حرفاً',
    'Document questions are temporarily unavailable.': 'الأسئلة عن المستند غير متاحة مؤقتاً.',
    'Reading the protected copy…': 'جارٍ قراءة النسخة المحمية…',
    'The document question could not be answered.': 'تعذرت الإجابة عن السؤال حول المستند.',
    'Response details': 'تفاصيل الإجابة',
    'Reveal value': 'إظهار القيمة',
    'No quick question': 'بدون سؤال سريع',
    'Select a quick question': 'اختر سؤالاً سريعاً',
    'A citation was adjusted to the matching policy rule': 'تمت مطابقة الإحالة مع قاعدة السياسة',
    'An unsupported rule reference was omitted. Use the matching rules below.': 'تم تجاهل مرجع غير موجود. راجع القواعد المطابقة أدناه.',
    'Policy guidance supports your review; it does not make the sharing decision.': 'تساعدك إرشادات السياسة في المراجعة، لكنها لا تتخذ قرار المشاركة.',
    'matching rules': 'قواعد مطابقة',
    'Matching rule': 'قاعدة مطابقة',
    'This answer is guidance for your review; it does not change a decision or document.': 'هذه الإجابة إرشادية ولا تغيّر قراراً أو مستنداً.',
    'Use these rules during review': 'استخدم هذه القواعد أثناء المراجعة',
    'Download audit file': 'تنزيل ملف التدقيق',
    'Prompts and document text are not stored or sent to charting services.': 'لا تُحفظ التعليمات أو نصوص المستندات ولا تُرسل إلى خدمات الرسوم البيانية.',
    'Values are masked. The table omits raw values, explanations, and replacement text.': 'القيم محجوبة. لا يتضمن الجدول القيم الأصلية أو التفسيرات أو البدائل.',
    'This trace shows the recorded action, source policy, and reviewer.': 'يعرض هذا السجل الإجراء المسجل والسياسة المرجعية والمراجع.',
    'The rule was saved.': 'تم حفظ القاعدة.',
    'The rule was deleted.': 'تم حذف القاعدة.',
    'Audit values are masked. Chart data stays local.': 'قيم التدقيق محجوبة. تبقى بيانات الرسوم على هذا الجهاز.',
    'Sharing activity': 'نشاط المشاركة',
    'The file type is not supported.': 'نوع الملف غير مدعوم.',
    'Unsupported file': 'ملف غير مدعوم',
    'Mean call latency': 'متوسط زمن الاستدعاء',
    'Document processing': 'معالجة المستند',
    'Purpose': 'الغرض',
    'Duration': 'المدة',
    'Input tokens': 'رموز الإدخال',
    'Output tokens': 'رموز الإخراج',
    'Tokens per second': 'رموز في الثانية',
    'JSON attempt': 'محاولة JSON',
    'JSON validation time': 'زمن التحقق من JSON',
    'Request reference': 'مرجع الطلب',
    'Recorded calls': 'الاستدعاءات المسجلة',
    'Model load time': 'زمن تحميل النموذج',
    'Question': 'السؤال',
    'Recorded sharing activity and decisions.': 'نشاط المشاركة والقرارات المسجلة.',
    'No matching sharing rule was cited.': 'لم تتم الإشارة إلى قاعدة مشاركة مطابقة.',
    'The proposed decision is below the confidence threshold.': 'مستوى الثقة في القرار المقترح أقل من الحد المطلوب.',
    'No matching sharing rule was verified; this item is held for review.': 'تعذر التحقق من قاعدة مشاركة مطابقة؛ أُحيل العنصر للمراجعة.',
    'A proposed substitute did not pass validation and was held for review.': 'لم يجتز البديل المقترح التحقق؛ أُحيل العنصر للمراجعة.',
    'An overlapping finding was excluded from the protected copy.': 'استُبعدت نتيجة متداخلة من النسخة المحمية.',
    'A scan result needed additional review; no unsupported decision was applied.': 'احتاجت نتيجة الفحص إلى مراجعة إضافية؛ لم يُطبّق قرار غير مدعوم.',
    'The document was processed in multiple sections.': 'تمت معالجة المستند على أجزاء متعددة.',
    'An invalid finding was ignored.': 'تم تجاهل نتيجة غير صالحة.',
    'A safeguard check was recorded.': 'تم تسجيل فحص لضوابط الحماية.',
    'Details': 'التفاصيل',
    'Search detail': 'تفاصيل البحث',
    'A policy answer is context for a reviewer, never an instruction to the release pipeline.': 'إجابة السياسة مرجع للمراجع، وليست توجيهاً لمسار المشاركة.',
    'Analysis is unavailable.': 'الفحص غير متاح حالياً.',
    'Ask in Arabic or English. Answers are based on the matching sharing rules.': 'اسأل بالعربية أو الإنجليزية. تستند الإجابات إلى قواعد المشاركة المطابقة.',
    'Choose whether the data is shared as-is, replaced, removed, or held for review.': 'اختر مشاركة البيانات كما هي، أو استبدالها، أو إزالتها، أو إحالتها للمراجعة.',
    'Embedding model': 'نموذج التضمين',
    'External privacy service': 'خدمة خصوصية خارجية',
    'I understand this changes the policy for every future analysis.': 'أفهم أن هذا التغيير سيُطبق على كل عمليات الفحص القادمة.',
    'On this device': 'على هذا الجهاز',
    'Policy search is unavailable right now. The sharing rules below remain available.': 'البحث في السياسة غير متاح حالياً، لكن قواعد المشاركة أدناه متاحة.',
    'Preparing the protected copy…': 'جارٍ تجهيز النسخة المحمية…',
    'Recipients': 'الجهات المستلمة',
    'Scanning in progress': 'جارٍ فحص المستند',
    'They were excluded from this protected copy.': 'تم استبعادها من هذه النسخة المحمية.',
    'This explanation appears to reviewers and in the audit record.': 'يظهر هذا التفسير للمراجعين وفي سجل التدقيق.',
    'Turn this off to show matching rules without a written summary.': 'أوقف هذا الخيار لعرض القواعد المطابقة دون ملخص مكتوب.',
    'items checked. Opening privacy review.': 'عناصر تم فحصها. جارٍ فتح مراجعة الخصوصية.',
    'system': 'النظام',
    'Employee records, payroll and HR processing': 'سجلات الموظفين والرواتب ومعالجة شؤون الموارد البشرية',
    'Repair or service work that requires the complaint and invoice details': 'أعمال الإصلاح أو الخدمة التي تتطلب تفاصيل الشكوى والفاتورة',
    'Internal analysis and reporting': 'التحليل الداخلي وإعداد التقارير',
    "Handling the customer's case and calling them back": 'معالجة حالة العميل والتواصل معه',
    'A receiver with no rules yet': 'جهة لم تُحدد لها قواعد بعد',
    "HR processes the employee's own record, so identifiers are allowed when they are needed for that processing.": 'تعالج الموارد البشرية سجل الموظف، ويُسمح بالمعرّفات عند الحاجة إليها لهذا الغرض.',
    'An outside company. It gets what it needs to do the job, and nothing that identifies a person.': 'جهة خارجية تحصل على ما يلزم للعمل فقط، دون بيانات تكشف هوية الأشخاص.',
    'Works with totals and trends. Needs business figures, not people.': 'يركز على الإجماليات والاتجاهات ومؤشرات العمل، لا على بيانات الأفراد.',
    'Talks to the customer directly, so contact details are allowed, but official identifiers are not.': 'يتواصل مع العميل مباشرة؛ لذلك قد يحتاج بيانات الاتصال دون المعرّفات الرسمية.',
    'No rules are defined for this receiver, so every sensitive item comes back as ASK_HUMAN. The system does not invent a rule.': 'لا توجد قواعد لهذه الجهة؛ لذلك تُحال كل معلومة حساسة إلى المراجعة البشرية.',
    'What may HR keep in an employee record?': 'ما البيانات التي يمكن للموارد البشرية الاحتفاظ بها في سجل الموظف؟',
    'هل يمكن مشاركة رقم الهوية مع مورد خارجي؟': 'هل يمكن مشاركة رقم الهوية مع مورد خارجي؟',
    'What does the analyst receive?': 'ما البيانات التي يحصل عليها المحلل؟',
    'What happens if no rule covers an item?': 'ماذا يحدث إذا لم تغطِ القواعد عنصراً؟',
    'No rule for this recipient answers the question.': 'لا توجد قاعدة لهذه الجهة تجيب عن السؤال.',
    'The policy has no matching rule. Sensitive data is not approved automatically.': 'لا توجد قاعدة مطابقة في السياسة. لا تتم الموافقة تلقائياً على البيانات الحساسة.',
    'The policy does not include a matching rule for this question.': 'لا تتضمن السياسة قاعدة مطابقة لهذا السؤال.',
    'Answer-writing is switched off: the rules above are the answer.': 'صياغة الإجابة متوقفة؛ القواعد أعلاه هي المرجع.',
    'The answer is based on the rules shown below.': 'تستند الإجابة إلى القواعد المعروضة أدناه.',
    'A policy citation needs review.': 'تحتاج إحالة السياسة إلى مراجعة.',
    'The model did not return a usable answer; the retrieved rules are shown instead. Nothing was invented on its behalf.': 'تعذرت صياغة إجابة موثوقة؛ تُعرض القواعد المسترجعة بدلاً منها.',
    'The answer cites no retrieved rule. Read the rules below rather than the summary.': 'لا تستند الإجابة إلى قاعدة مسترجعة. يرجى مراجعة القواعد أدناه.',
    'No rule in the policy matches this question. That is an answer: the policy is silent, so the item goes to a person.': 'لا توجد قاعدة في السياسة تطابق السؤال؛ تُحال المسألة إلى المراجعة البشرية.',
    'No rule for {recipient} answers this. A policy with no rule for the item is not permission - the item reaches a person as ASK_HUMAN.': 'لا توجد قاعدة للجهة {recipient} تجيب عن السؤال. غياب القاعدة لا يعني السماح؛ تُحال المسألة للمراجعة.',
}

ACTION_LABELS = {
    "KEEP": ("Share as-is", "مشاركة كما هي"),
    "REPLACE": ("Use a safe substitute", "استخدام بديل آمن"),
    "REMOVE": ("Remove from shared copy", "إزالة من النسخة المشتركة"),
    "ASK_HUMAN": ("Needs your review", "تحتاج إلى مراجعتك"),
    "REJECT_SUGGESTION": ("Keep out of shared copy", "استبعاد من النسخة المشتركة"),
}
PII_LABELS = {
    "PERSON_NAME": ("Person name", "اسم شخص"),
    "NATIONAL_ID": ("National ID", "رقم هوية وطنية"),
    "PHONE": ("Phone number", "رقم هاتف"), "EMAIL": ("Email address", "بريد إلكتروني"),
    "ADDRESS": ("Address", "عنوان"), "BANK_ACCOUNT": ("Bank account", "حساب بنكي"),
    "IBAN": ("IBAN", "رقم آيبان"), "DATE_OF_BIRTH": ("Date of birth", "تاريخ الميلاد"),
    "PASSPORT_NUMBER": ("Passport number", "رقم جواز سفر"),
    "CUSTOMER_ID": ("Customer ID", "رقم عميل"), "EMPLOYEE_ID": ("Employee ID", "رقم موظف"),
    "COMPANY_IDENTIFIER": ("Company identifier", "معرّف شركة"),
}
def _is_arabic() -> bool:
    return st.session_state.get("ui_language", "en") == "ar"


def _t(text: str) -> str:
    return ARABIC_TRANSLATIONS.get(text, text) if _is_arabic() else text


def _action_label(action: str) -> str:
    labels = ACTION_LABELS.get(action)
    if labels:
        return labels[1] if _is_arabic() else labels[0]
    return _t(action)


def _pii_label(pii_type: str) -> str:
    labels = PII_LABELS.get(pii_type)
    if labels:
        return labels[1] if _is_arabic() else labels[0]
    return pii_type


def _policy_note(note: str) -> str:
    """Translate policy-search statuses without exposing provider exception text."""
    if note.startswith("No rule for ") or note.startswith("No rule in the policy"):
        return _t("The policy does not include a matching rule for this question.")
    if note.startswith("Answer-writing is switched off"):
        return _t("Answer-writing is switched off: the rules above are the answer.")
    if note.startswith("The retrieved rules are shown unsummarised"):
        return _t("The answer is based on the rules shown below.")
    if note.startswith(("The model could not write", "The model did not return",
                        "The answer cites no retrieved rule")):
        return _t("The answer is based on the rules shown below.")
    if note.startswith(("The model wrote", "The model cited rule")):
        return _t("A policy citation needs review.")
    return _t(note)


def _guardrail_note(note: str) -> str:
    """Show a localized, value-safe summary of a guardrail event."""
    if "cited rule" in note or "valid rule" in note or "without a valid rule" in note:
        return _t("No matching sharing rule was verified; this item is held for review.")
    if "substitute" in note:
        return _t("A proposed substitute did not pass validation and was held for review.")
    if "overlapped" in note:
        return _t("An overlapping finding was excluded from the protected copy.")
    if "second look" in note or "usable JSON" in note:
        return _t("A scan result needed additional review; no unsupported decision was applied.")
    if "document was analysed in" in note:
        return _t("The document was processed in multiple sections.")
    if "finding was not an object" in note:
        return _t("An invalid finding was ignored.")
    return _t("A safeguard check was recorded.")


def _review_reason(reason: str) -> str:
    if reason.startswith("no rule"):
        return _t("No matching sharing rule was cited.")
    if reason.startswith("the model's confidence"):
        return _t("The proposed decision is below the confidence threshold.")
    return _t(reason)


def _apply_theme() -> None:
    """The visual system for the privacy intelligence lab."""
    st.markdown("""
    <style>
        :root {
            color-scheme: dark;
            --ink:#090a0c; --panel:#111316; --panel-raised:#17191d;
            --line:rgba(228,231,236,.12); --muted:#969ba4; --paper:#f2f0eb;
            --coral:#ff665d; --coral-hot:#ff8a73; --mint:#a7e5cc; --amber:#f2bd79;
            --display:Georgia,"Times New Roman",serif;
            --ui:Bahnschrift,"Segoe UI Variable","Segoe UI",sans-serif;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 72% 1%,rgba(255,102,93,.105),transparent 29%),
                radial-gradient(ellipse at 8% 38%,rgba(107,126,111,.075),transparent 31%),
                linear-gradient(145deg,#0b0c0e 0%,#101114 47%,#0b0c0e 100%);
            color:var(--paper); font-family:var(--ui),sans-serif;
        }
        .stApp::before {
            content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
            opacity:.15; background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
            linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
            background-size:48px 48px; mask-image:linear-gradient(to bottom,black,transparent 78%);
        }
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:transparent; }
        [data-testid="stHeader"] { height:2.5rem; }
        [data-testid="stMain"] { position:relative; z-index:1; }
        .main .block-container { max-width:1560px; padding:3rem 3.1rem 4.5rem; }
        .brand-lockup { display:flex; align-items:baseline; gap:.55rem; min-height:40px; color:var(--paper); }
        .brand-latin { font:700 1.55rem/1 var(--ui),sans-serif; }
        .brand-divider { color:var(--coral-hot); font:400 1.1rem/1 var(--ui),sans-serif; }
        .brand-arabic { color:#f3eee6; font:600 1.48rem/1 var(--display),serif; }
        .brand-tagline { margin-top:.55rem; color:#aaa8a2; font-size:.88rem; line-height:1.45; }
        [data-testid="stSidebar"] {
            background:linear-gradient(180deg,#111214 0%,#0c0d0f 100%);
            border-right:1px solid var(--line);
        }
        [data-testid="stSidebar"] .brand-lockup {
            gap:.65rem; min-height:58px; animation:sitr-brand-glow 4s ease-in-out infinite;
        }
        [data-testid="stSidebar"] .brand-latin { font-size:36px; font-weight:900; }
        [data-testid="stSidebar"] .brand-divider { font-size:1.4rem; font-weight:600; }
        [data-testid="stSidebar"] .brand-arabic { font-size:36px; font-weight:900; }
        @keyframes sitr-brand-glow {
            0%,100% { filter:drop-shadow(0 0 4px rgba(255,138,115,.2)); }
            50% { filter:drop-shadow(0 0 14px rgba(255,138,115,.5)); }
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            min-height:42px; padding:.56rem .72rem; border:1px solid transparent;
            border-radius:7px; color:#b5b5b2; transition:background .2s,border-color .2s,color .2s;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { color:var(--paper); background:#191a1d; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            color:#fff; background:linear-gradient(100deg,rgba(255,102,93,.14),rgba(255,102,93,.025));
            border-color:rgba(255,102,93,.2);
        }
        h1,h2,h3,[data-testid="stMetricValue"] { color:var(--paper) !important; font-family:var(--ui),sans-serif !important; }
        h1 { font-family:var(--display),serif !important; font-size:2.85rem !important; font-weight:400 !important; }
        h2 { font-family:var(--display),serif !important; font-size:2rem !important; font-weight:400 !important; }
        h3 { font-size:1.15rem !important; font-weight:600 !important; }
        p, label, [data-testid="stCaptionContainer"] { color:#c0bfbb; }
        [data-testid="stMetric"] {
            min-height:116px; padding:20px 22px; border:1px solid var(--line); border-radius:8px;
            background:linear-gradient(145deg,rgba(31,33,36,.86),rgba(17,18,20,.94));
            box-shadow:0 18px 50px rgba(0,0,0,.17);
        }
        [data-testid="stMetricLabel"] { color:#aaa9a3 !important; font-size:.84rem !important; }
        [data-testid="stMetricValue"] { font-size:1.62rem !important; font-weight:500 !important; }
        [data-testid="stMetricDelta"] { font-size:.78rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color:var(--line) !important; border-radius:8px !important;
            background:linear-gradient(148deg,rgba(28,30,33,.77),rgba(15,16,18,.86)) !important;
            box-shadow:0 16px 48px rgba(0,0,0,.17);
            transition:border-color .24s,transform .24s,background .24s;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:rgba(255,102,93,.28) !important; }
        [data-testid="stFileUploader"] section {
            min-height:210px; background:linear-gradient(135deg,rgba(33,35,38,.87),rgba(18,19,21,.94));
            border:1px dashed rgba(232,226,215,.28); border-radius:8px; transition:border-color .2s,background .2s;
        }
        [data-testid="stFileUploader"] section:hover { border-color:var(--coral); background:rgba(255,102,93,.055); }
        .hero { padding:1.4rem 0 1.8rem; margin:.2rem 0 1.2rem; border-bottom:1px solid var(--line); }
        .hero-kicker,.section-label,.eyebrow {
            color:var(--coral-hot); font-family:var(--ui),sans-serif; font-size:.68rem;
            font-weight:700; text-transform:uppercase; letter-spacing:0;
        }
        .hero-title { margin:.5rem 0; color:var(--paper); font:400 2.7rem/1.12 var(--display),serif; }
        .hero-copy { max-width:760px; margin:0; color:#b8b6af; font-size:1rem; line-height:1.7; }
        .section-label { margin:.5rem 0 .65rem; color:#b2afa8; }
        .lab-hero {
            position:relative; display:grid; grid-template-columns:minmax(0,1.2fr) minmax(280px,.8fr);
            min-height:380px; overflow:hidden; margin:0 0 1.5rem; padding:2.9rem 3.15rem;
            align-items:center; isolation:isolate; border:1px solid rgba(235,231,224,.15); border-radius:10px;
            background:linear-gradient(112deg,rgba(24,25,27,.98) 0%,rgba(18,19,21,.94) 49%,rgba(33,23,23,.86) 100%);
            box-shadow:0 28px 100px rgba(0,0,0,.27),inset 0 1px rgba(255,255,255,.055);
        }
        .lab-hero::before {
            content:""; position:absolute; z-index:-1; width:560px; height:560px; inset-inline-end:-100px; top:-145px;
            border-radius:50%; background:radial-gradient(circle,rgba(255,102,93,.14),rgba(255,102,93,.025) 34%,transparent 68%);
        }
        .lab-hero::after {
            content:""; position:absolute; z-index:-1; inset:0; opacity:.26; pointer-events:none;
            background:linear-gradient(112deg,transparent 48%,rgba(255,255,255,.035) 48.1%,transparent 48.3%);
        }
        .lab-copy { position:relative; z-index:2; max-width:660px; }
        .lab-brand {
            display:flex; align-items:center; gap:.1em; width:max-content; max-width:100%;
            margin-bottom:.8rem; padding:0 0 .3rem; border-bottom:1px solid rgba(255,138,115,.3);
            color:#f2eee7; line-height:1;
        }
        .lab-brand .brand-latin {
            font:900 3rem/1 var(--ui),sans-serif; letter-spacing:-.055em;
            background:linear-gradient(110deg,#fff8ef 12%,#ffad96 100%);
            background-clip:text; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        }
        .lab-brand .brand-divider {
            color:var(--coral-hot); font-size:2.25rem; font-weight:500;
            text-shadow:0 0 16px rgba(255,102,93,.4);
        }
        .lab-brand .brand-arabic {
            color:#f5eee6; font:700 2.65rem/1 var(--display),serif;
            text-shadow:0 0 18px rgba(255,138,115,.15);
        }
        .lab-overline { display:flex; gap:10px; align-items:center; color:#b6b0a7; font-size:.78rem; text-transform:uppercase; }
        .signal-dot { width:7px; height:7px; border-radius:50%; background:var(--mint); box-shadow:0 0 14px rgba(167,229,204,.75); }
        .lab-title { max-width:630px; margin:1.1rem 0 1rem; color:#f3f0e9; font:400 3.35rem/1.04 var(--display),serif; }
        .lab-title span { color:var(--coral-hot); font-style:italic; }
        .lab-summary { max-width:500px; color:#b9b6ae; font-size:1rem; line-height:1.7; }
        .lab-meta { display:flex; gap:30px; flex-wrap:wrap; margin-top:1.8rem; }
        .lab-meta-item { display:grid; gap:4px; }
        .lab-meta-label { color:#858680; font-size:.7rem; text-transform:uppercase; }
        .lab-meta-value { color:#e5e1d9; font-size:.88rem; }
        .shield-stage { position:relative; display:grid; place-items:center; height:300px; perspective:850px; }
        .shield-aura {
            position:absolute; width:220px; aspect-ratio:1; border-radius:50%;
            background:radial-gradient(circle,rgba(255,102,93,.17),rgba(255,102,93,.055) 37%,transparent 69%);
            filter:blur(4px); animation:aura-breathe 5s ease-in-out infinite;
        }
        .orbit { position:absolute; width:268px; height:122px; border:1px solid rgba(255,142,124,.22); border-radius:50%; transform:rotate(-27deg); }
        .orbit::after { content:""; position:absolute; top:13px; left:17%; width:5px; height:5px; border-radius:50%; background:#ffb095; box-shadow:0 0 13px #ff806d; }
        .orbit.orbit-two { width:230px; height:150px; transform:rotate(57deg); border-color:rgba(167,229,204,.18); }
        .orbit.orbit-two::after { top:auto; bottom:9px; left:auto; right:13%; background:var(--mint); box-shadow:0 0 12px var(--mint); }
        .shield-backplate {
            position:absolute; width:145px; height:174px; transform:translate(9px,8px) rotateY(-13deg) rotateX(8deg);
            clip-path:polygon(50% 0,95% 16%,88% 62%,50% 100%,12% 62%,5% 16%);
            background:linear-gradient(145deg,rgba(255,152,130,.88),rgba(96,35,36,.56) 42%,rgba(43,64,58,.72));
            filter:drop-shadow(0 0 30px rgba(255,102,93,.3));
        }
        .shield-shell {
            position:relative; display:grid; place-items:center; width:142px; height:171px;
            transform:rotateY(13deg) rotateX(8deg); clip-path:polygon(50% 0,95% 16%,88% 62%,50% 100%,12% 62%,5% 16%);
            background:linear-gradient(142deg,rgba(255,238,220,.9) 0%,rgba(255,102,93,.92) 18%,rgba(104,48,50,.95) 54%,rgba(132,183,159,.74) 100%);
            animation:shield-hover 6s ease-in-out infinite;
            filter:drop-shadow(0 22px 26px rgba(0,0,0,.5));
        }
        .shield-inset {
            position:absolute; inset:3px; clip-path:inherit; display:grid; place-items:center;
            background:linear-gradient(150deg,rgba(42,45,47,.98),rgba(19,22,23,.98) 61%,rgba(24,37,33,.98));
            box-shadow:inset 0 0 32px rgba(255,102,93,.2);
        }
        .shield-inset::before { content:""; position:absolute; inset:9px; clip-path:inherit; border:1px solid rgba(255,173,148,.27); }
        .shield-glyph {
            position:relative; width:44px; height:48px; transform:translateY(-2px);
            clip-path:polygon(50% 0,100% 18%,91% 65%,50% 100%,9% 65%,0 18%);
            background:linear-gradient(145deg,#ffd3c2,rgba(255,102,93,.62));
        }
        .shield-glyph::before {
            content:""; position:absolute; inset:2px;
            clip-path:polygon(50% 0,100% 18%,91% 65%,50% 100%,9% 65%,0 18%);
            background:linear-gradient(150deg,#30312f,#171a19 70%);
        }
        .shield-glyph::after {
            content:""; position:absolute; width:18px; height:9px; left:12px; top:19px;
            border-left:2px solid var(--mint); border-bottom:2px solid var(--mint); transform:rotate(-45deg);
            filter:drop-shadow(0 0 5px rgba(167,229,204,.8));
        }
        .shield-scan {
            position:absolute; z-index:2; left:12%; right:12%; top:18%; height:1px; opacity:.72;
            background:linear-gradient(90deg,transparent,rgba(255,196,173,.92),transparent);
            box-shadow:0 0 12px rgba(255,116,98,.75); animation:none; opacity:.24;
        }
        .shield-stage.is-scanning .shield-scan { animation:scan-sweep 1.1s ease-in-out infinite; height:2px; opacity:1; }
        .scan-live { position:relative; display:flex; align-items:center; gap:12px; overflow:hidden; padding:12px 14px; border:1px solid rgba(255,102,93,.19); border-radius:6px; color:#e8dfd5; background:rgba(255,102,93,.045); font-size:.78rem; }
        .scan-live-beam { display:block; flex:0 0 38px; height:2px; background:linear-gradient(90deg,transparent,var(--coral-hot),transparent); box-shadow:0 0 12px rgba(255,102,93,.65); animation:scan-live-sweep 1.5s ease-in-out infinite; }
        @keyframes scan-live-sweep { 0%,100% { transform:translateX(-8px); opacity:.5; } 50% { transform:translateX(8px); opacity:1; } }
        .shield-readout {
            position:absolute; inset-inline-end:1%; top:14%; min-width:158px; padding:13px 15px;
            border:1px solid rgba(232,231,224,.15); border-radius:7px; color:#ddd8ce;
            background:rgba(27,28,29,.74); box-shadow:0 14px 35px rgba(0,0,0,.3); backdrop-filter:blur(14px);
            font-size:.67rem; animation:readout-float 7s ease-in-out infinite;
        }
        .shield-readout.bottom { top:auto; inset-inline-start:9%; inset-inline-end:auto; bottom:12%; animation-delay:-3s; }
        .readout-label { display:block; margin-bottom:6px; color:#98968f; font-size:.64rem; text-transform:uppercase; }
        .readout-value { color:#efebe2; font-size:.75rem; }
        .readout-value .online { color:var(--mint); }
        .lab-footnote { display:flex; align-items:center; gap:9px; margin-top:1.3rem; color:#777973; font-size:.68rem; }
        .workflow-shell {
            position:relative; padding:21px 23px 20px; margin:0 0 1.7rem; overflow:hidden;
            border:1px solid var(--line); border-radius:8px;
            background:linear-gradient(105deg,rgba(24,25,27,.94),rgba(16,17,19,.92));
        }
        .workflow-heading { display:flex; align-items:center; justify-content:space-between; margin-bottom:17px; }
        .workflow-caption { color:#e0dcd4; font-size:.8rem; }
        .workflow-state { color:#aaa79f; font-size:.72rem; text-transform:uppercase; }
        .pipeline { position:relative; display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; }
        .pipeline::before { content:""; position:absolute; top:13px; left:2%; right:2%; height:1px; background:rgba(239,232,220,.13); }
        .workflow-step { position:relative; display:grid; gap:8px; justify-items:start; min-width:0; color:#777974; }
        .workflow-node { display:grid; place-items:center; width:27px; height:27px; z-index:1; border:1px solid #434448; border-radius:50%; background:#17181a; font-size:.59rem; }
        .workflow-name { overflow:hidden; max-width:100%; color:#aaa9a2; font-size:.76rem; line-height:1.4; text-overflow:ellipsis; white-space:nowrap; }
        .workflow-step.is-complete .workflow-node { color:#161a18; border-color:var(--mint); background:var(--mint); }
        .workflow-step.is-complete .workflow-name { color:#c5d9ce; }
        .workflow-step.is-current .workflow-node { color:#fff; border-color:var(--coral); background:var(--coral); box-shadow:0 0 0 5px rgba(255,102,93,.11),0 0 20px rgba(255,102,93,.38); }
        .workflow-step.is-current .workflow-name { color:#fff; }
        .workflow-step.is-current.is-scanning .workflow-node { animation:node-pulse 1.5s ease-in-out infinite; }
        .card-title { color:#f2f0eb; font-weight:650; font-size:.98rem; }
        .muted { color:var(--muted); font-size:.8rem; }
        .receiver-profile { min-height:185px; padding:4px 0 8px; }
        .profile-head { display:flex; align-items:center; gap:13px; margin-bottom:16px; }
        .profile-seal {
            display:grid; place-items:center; flex:0 0 39px; height:39px; border:1px solid rgba(232,229,221,.17);
            clip-path:polygon(50% 0,96% 17%,87% 64%,50% 100%,13% 64%,4% 17%);
            color:#d4cfc4; background:linear-gradient(145deg,#282a2c,#151719); font-size:.71rem;
        }
        .receiver-profile.is-selected .profile-seal { color:#fff; border-color:var(--coral); background:linear-gradient(145deg,#b74943,#351f20); }
        .profile-name { color:#efede6; font-family:var(--display),serif; font-size:1.25rem; line-height:1.2; }
        .profile-code { margin-top:4px; color:#858680; font-size:.59rem; }
        .profile-description { min-height:42px; color:#b5b3ac; font-size:.85rem; line-height:1.55; }
        .profile-rule-line { display:flex; justify-content:space-between; gap:8px; margin:13px 0 8px; color:#c1beb6; font-size:.74rem; }
        .profile-rule-line span:last-child { color:#e9e4db; }
        .profile-bar { display:flex; gap:2px; height:4px; overflow:hidden; border-radius:4px; background:#2b2c2e; }
        .profile-action-list { display:flex; flex-wrap:wrap; gap:5px 10px; margin-top:8px; }
        .profile-action { color:#b5b2ab; font-size:.68rem; }
        .profile-action b { font-weight:600; }
        .rule-card { padding:1rem 1.05rem; border:1px solid var(--line); border-radius:7px; background:linear-gradient(145deg,#1b1d1f,#131416); margin:.35rem 0 .6rem; }
        .rule-id { color:var(--coral-hot); font:600 .74rem var(--ui),sans-serif; }
        .rule-reason { color:#d0cec7; margin:.45rem 0; }
        .score-chip { color:#b8b5ad; font-size:.76rem; }
        .action-chip { display:inline-block; padding:.24rem .55rem; border:1px solid rgba(255,255,255,.09); border-radius:4px; font-size:.76rem; font-weight:650; background:#252628; color:#e4e1da; }
        div.stButton > button[kind="primary"] {
            min-height:43px; border:1px solid rgba(255,139,120,.25); border-radius:6px;
            color:#fff7f2; font-family:var(--ui),sans-serif; font-weight:700;
            background:linear-gradient(105deg,#d94f49,#fb6c5e 58%,#e75a50);
            box-shadow:0 10px 28px rgba(231,80,71,.2); transition:transform .2s,box-shadow .2s,filter .2s;
        }
        div.stButton > button[kind="primary"]:hover { transform:translateY(-2px); filter:brightness(1.08); box-shadow:0 14px 34px rgba(231,80,71,.32); }
        div.stButton > button { min-height:40px; border-radius:6px; border-color:rgba(230,226,216,.18); transition:transform .18s,border-color .18s,background .18s; }
        div.stButton > button:hover { transform:translateY(-1px); border-color:rgba(255,126,108,.55); }
        div[data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--line); }
        button[data-baseweb="tab"] { min-height:42px; border-radius:5px 5px 0 0; color:#aaa8a1; }
        button[data-baseweb="tab"][aria-selected="true"] { color:#fff !important; }
        [data-testid="stExpander"] { border-color:var(--line); border-radius:7px; background:rgba(24,25,27,.55); }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:7px; overflow:hidden; }
        .stProgress > div > div { background:linear-gradient(90deg,var(--coral),#ffc08a); }
        .pii-reveal { position:relative; display:inline-flex; align-items:center; margin:0 2px; vertical-align:baseline; cursor:pointer; }
        .pii-reveal input { position:absolute; width:1px; height:1px; opacity:0; }
        .pii-mask { display:inline-block; padding:0 6px; border:1px solid rgba(255,102,93,.38); border-radius:3px; color:#ffb5a9; background:rgba(255,102,93,.13); font:600 .68rem var(--ui),sans-serif; line-height:1.7; }
        .pii-value { display:none; padding:0 4px; border-bottom:1px solid var(--coral); color:#fff0e9; background:rgba(255,102,93,.13); animation:reveal-in .24s ease-out; }
        .pii-reveal input:checked ~ .pii-mask { display:none; }
        .pii-reveal input:checked ~ .pii-value { display:inline-block; }
        .pii-reveal input:focus-visible + .pii-mask { outline:2px solid #ffe1d2; outline-offset:3px; }
        .decision-timeline { position:relative; display:grid; gap:0; margin:.35rem 0 1.2rem; padding:4px 0 4px 20px; padding-inline-start:20px; }
        .decision-timeline::before { content:""; position:absolute; inset-inline-start:5px; top:12px; bottom:12px; width:1px; background:linear-gradient(var(--coral),rgba(255,102,93,.08)); }
        .timeline-entry { position:relative; display:grid; grid-template-columns:minmax(150px,.7fr) minmax(0,2fr) auto; gap:18px; align-items:center; padding:16px 18px; border-bottom:1px solid rgba(233,230,222,.075); background:linear-gradient(90deg,rgba(255,255,255,.025),transparent); }
        .timeline-entry::before { content:""; position:absolute; inset-inline-start:-19px; width:9px; height:9px; border:2px solid var(--coral); border-radius:50%; background:#17181a; box-shadow:0 0 13px rgba(255,102,93,.34); }
        .timeline-date { color:#aaa79f; font-size:.76rem; }
        .timeline-document { overflow:hidden; color:#efede6; font-family:var(--display),serif; font-size:1.08rem; text-overflow:ellipsis; white-space:nowrap; }
        .timeline-detail { color:#a6a39c; font-size:.75rem; }
        .timeline-run { color:#dfb5a9; font-size:.72rem; white-space:nowrap; }
        @keyframes shield-hover { 0%,100% { transform:translateY(0) rotateY(13deg) rotateX(8deg); } 50% { transform:translateY(-8px) rotateY(17deg) rotateX(5deg); } }
        @keyframes scan-sweep { 0%,100% { top:17%; opacity:.2; } 50% { top:78%; opacity:.95; } }
        @keyframes aura-breathe { 0%,100% { transform:scale(.94); opacity:.72; } 50% { transform:scale(1.08); opacity:1; } }
        @keyframes readout-float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-6px); } }
        @keyframes node-pulse { 0%,100% { box-shadow:0 0 0 4px rgba(255,102,93,.08),0 0 12px rgba(255,102,93,.24); } 50% { box-shadow:0 0 0 8px rgba(255,102,93,.04),0 0 22px rgba(255,102,93,.43); } }
        @keyframes reveal-in { from { opacity:0; filter:blur(4px); } to { opacity:1; filter:blur(0); } }
        @media (max-width: 1050px) {
            .main .block-container { padding:2.7rem 1.7rem 4rem; }
            .lab-hero { display:block; min-height:355px; padding:2rem; padding-inline-end:185px; }
            .lab-copy { max-width:350px; }
            .lab-title { max-width:350px; font-size:2.5rem; }
            .lab-summary { max-width:320px; }
            .lab-meta { gap:13px; margin-top:1.1rem; }
            .shield-stage { position:absolute; z-index:0; top:50%; inset-inline-end:-42px; width:250px; height:285px; transform:translateY(-50%); opacity:.76; pointer-events:none; }
            .shield-readout { display:none; }
        }
        @media (max-width: 720px) {
            .main .block-container { padding:2rem 1rem 3rem; }
            .lab-hero { grid-template-columns:1fr; min-height:0; padding:1.7rem 1.35rem 1rem; }
            .lab-brand .brand-latin { font-size:2.6rem; }
            .lab-brand .brand-divider { font-size:1.95rem; }
            .lab-brand .brand-arabic { font-size:2.3rem; }
            .lab-title { max-width:500px; margin-top:1rem; font-size:2.45rem; }
            .lab-summary { font-size:.85rem; }
            .lab-meta { gap:18px; margin-top:1.1rem; }
            .shield-stage { position:relative; top:auto; inset-inline-end:auto; width:auto; height:240px; margin-top:-3px; transform:none; opacity:1; pointer-events:auto; }
            .shield-readout { display:block; }
            .shield-shell { transform:scale(.84) rotateY(13deg) rotateX(8deg); }
            .shield-backplate { transform:scale(.84) translate(9px,8px) rotateY(-13deg) rotateX(8deg); }
            .orbit { transform:scale(.82) rotate(-27deg); }
            .orbit.orbit-two { transform:scale(.82) rotate(57deg); }
            .shield-readout { inset-inline-end:0; top:8%; }
            .shield-readout.bottom { inset-inline-start:0; bottom:7%; }
            .pipeline { grid-template-columns:repeat(2,minmax(0,1fr)); row-gap:17px; }
            .pipeline::before { display:none; }
            .workflow-name { white-space:normal; min-height:2em; }
            .timeline-entry { grid-template-columns:1fr auto; gap:4px 10px; padding:12px 9px; }
            .timeline-date { grid-column:1; }
            .timeline-document { grid-column:1; }
            .timeline-detail { grid-column:1; }
            .timeline-run { grid-column:2; grid-row:1 / span 3; }
            .hero-title { font-size:2rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; }
        }
    </style>
    """, unsafe_allow_html=True)
    direction = "rtl" if _is_arabic() else "ltr"
    alignment = "right" if _is_arabic() else "left"
    st.markdown(
        f"<style>.stApp {{ direction:{direction}; text-align:{alignment}; }}"
        f"[data-testid='stMarkdownContainer'], [data-testid='stSidebar'] {{"
        f"direction:{direction}; text-align:{alignment}; }}"
        f"[data-testid='stHorizontalBlock'] {{ direction:{direction}; }}"
        f"[data-testid='stFileUploader'] {{ direction:{direction}; text-align:{alignment}; }}"
        f".lab-hero, .workflow-shell {{ direction:{direction}; }}</style>",
        unsafe_allow_html=True)


def _hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="hero" dir="{"rtl" if _is_arabic() else "ltr"}">'
        f'<div class="hero-kicker">{html.escape(_t(kicker))}</div>'
        f'<div class="hero-title">{html.escape(_t(title))}</div>'
        f'<p class="hero-copy">{html.escape(_t(copy))}</p></div>',
        unsafe_allow_html=True)


def _workflow_markup(active_step: int, scanning: bool = False) -> str:
    """Show pipeline state from the current analysis session, never estimated progress."""
    steps = ("Add document", "Scan", "Review findings", "Protect document", "Release")
    markup = []
    for index, label in enumerate(steps):
        state = "is-complete" if index < active_step or active_step >= len(steps) else ""
        if index == active_step and active_step < len(steps):
            state = "is-current" + (" is-scanning" if scanning else "")
        node = "&#10003;" if index < active_step else f"{index + 1:02d}"
        markup.append(
            f'<div class="workflow-step {state}"><span class="workflow-node">'
            f'{node}'
            f'</span><span class="workflow-name">{html.escape(_t(label))}</span></div>')
    return '<div class="pipeline">' + "".join(markup) + '</div>'


def _lab_hero_markup(receiver_label: str, rule_count: int, scanning: bool) -> str:
    scan_class = "is-scanning" if scanning else ""
    return (
        f'<section class="lab-hero {scan_class}" dir="{"rtl" if _is_arabic() else "ltr"}"><div class="lab-copy">'
        f'<div class="lab-brand" dir="ltr" aria-label="{html.escape(PRODUCT_BRAND)}">'
        '<span class="brand-latin">Sitr</span><span class="brand-divider">|</span>'
        '<bdi class="brand-arabic" dir="rtl">سِتر</bdi></div>'
        f'<div class="lab-overline"><span class="signal-dot"></span>{html.escape(_t("PRIVACY INTELLIGENCE LAB"))}</div>'
    f'<div class="lab-title">{html.escape(_t("A perimeter"))}<br>{html.escape(_t("for every release."))}</div>'
    f'<div class="lab-summary">{html.escape(_t("Inspect sensitive data. Apply recipient-specific rules. Approve a release with evidence attached."))}</div>'
        '<div class="lab-meta">'
    f'<div class="lab-meta-item"><span class="lab-meta-label">{html.escape(_t("Recipient"))}</span>'
        f'<span class="lab-meta-value">{html.escape(receiver_label)}</span></div>'
    f'<div class="lab-meta-item"><span class="lab-meta-label">{html.escape(_t("Supported formats"))}</span>'
    '<span class="lab-meta-value">PDF · DOCX · TXT</span></div>'
    f'<div class="lab-meta-item"><span class="lab-meta-label">{html.escape(_t("Privacy controls"))}</span>'
    f'<span class="lab-meta-value">{html.escape(_t("Review and audit"))}</span></div>'
        '</div></div>'
    f'<div class="shield-stage {scan_class}" role="img" aria-label="{html.escape(_t("Privacy shield, ready"))}">'
        '<div class="shield-aura"></div><div class="orbit"></div>'
        '<div class="orbit orbit-two"></div><div class="shield-backplate"></div>'
        '<div class="shield-shell"><div class="shield-inset">'
        '<div class="shield-glyph"></div><div class="shield-scan"></div>'
        '</div></div>'
    f'<div class="shield-readout"><span class="readout-label">{html.escape(_t("Private data"))}</span>'
    f'<span class="readout-value">{html.escape(_t("Protected release"))}</span></div>'
    f'<div class="shield-readout bottom"><span class="readout-label">{html.escape(_t("Recipient"))}</span>'
    f'<span class="readout-value">{html.escape(receiver_label)}</span><br>'
    f'<span class="readout-value">{rule_count} {html.escape(_t("active safeguards"))}</span></div></div></section>')


def _recipient_profile_markup(key: str, receiver_info, rules: list, selected: bool) -> str:
    action_counts = {action.value: 0 for action in Action}
    for rule in rules:
        action_counts[rule.action] = action_counts.get(rule.action, 0) + 1
    active_actions = [(action, count) for action, count in action_counts.items() if count]
    total_rules = len(rules)
    action_bar = "".join(
        f'<span style="width:{count / total_rules * 100:.2f}%;'
        f'background:{ACTION_STYLE[action][0]}"></span>'
        for action, count in active_actions) if total_rules else ""
    action_labels = []
    for action, count in active_actions:
        action_labels.append(
            f'<span class="profile-action"><b style="color:{ACTION_STYLE[action][0]}">'
            f'{html.escape(_action_label(action))}</b> {count}</span>')
    action_summary = "".join(action_labels) or \
        f'<span class="profile-action">{html.escape(_t("No sharing rules yet"))}</span>'
    return (
        f'<div class="receiver-profile {"is-selected" if selected else ""}">'
        '<div class="profile-head">'
        f'<div class="profile-seal">{html.escape(_t(receiver_info.label)[:1])}</div>'
        '<div><div class="profile-name">'
        f'{html.escape(_t(receiver_info.label))}</div></div></div>'
        f'<div class="profile-description">{html.escape(_t(receiver_info.description))}</div>'
        f'<div class="profile-rule-line"><span>{html.escape(_t("POLICY PROFILE"))}</span>'
        f'<span>{total_rules} {html.escape(_t("live rules"))}</span></div>'
        f'<div class="profile-bar">{action_bar}</div>'
        f'<div class="profile-action-list">{action_summary}</div></div>')


def _chart(fig, key: str, height: int = 300) -> None:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#bac8da", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#172337", bordercolor="#344963"))
    fig.update_xaxes(gridcolor="rgba(104,129,160,.13)", zerolinecolor="rgba(104,129,160,.13)")
    fig.update_yaxes(gridcolor="rgba(104,129,160,.13)", zerolinecolor="rgba(104,129,160,.13)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key=key)


def _mask_audit_value(value: object) -> str:
    """Mask again at the UI boundary, including short or malformed stored values."""
    text = str(value or "")
    masked = []
    for token in text.split():
        if len(token) <= 2:
            masked.append("•••")
        else:
            masked.append(token[0] + "*" * (len(token) - 2) + token[-1])
    return " ".join(masked)


# --------------------------------------------------------------------- rendering


def _mark(text: str, style: tuple, title: str) -> str:
    """A piece of the released text, highlighted, with the reason on hover."""
    return (f"<span title='{html.escape(title)}' style='background:{style[1]};"
            f"color:{style[0]};padding:0 2px;border-radius:3px;"
            f"border-bottom:2px solid {style[0]}'>{html.escape(text)}</span>")


def _reveal_mark(text: str, pii_type: str, title: str) -> str:
    """Render an original sensitive span as a click-to-reveal control."""
    return (
        f'<label class="pii-reveal" title="{html.escape(title, quote=True)}">'
        f'<input type="checkbox" aria-label="{html.escape(_t("Reveal value"), quote=True)} · '
        f'{html.escape(_pii_label(pii_type), quote=True)}">'
        '<span class="pii-mask" aria-hidden="true">REDACTED</span>'
        f'<span class="pii-value">{html.escape(text)}</span></label>')


def _gap(style: tuple, title: str) -> str:
    """Where a value was taken out of the release.

    A mark, not characters: what was removed is genuinely absent from the safe version, and
    this only shows where it was. It adds nothing to the text.
    """
    return (f"<span title='{html.escape(title)}' style='display:inline-block;width:3px;"
            f"height:0.9em;margin:0 4px;vertical-align:-0.1em;border-radius:1px;"
            f"background:{style[0]}'></span>")


def highlight_block(text: str, items: list[dict]) -> str:
    """One block of the original, with every item the model reported marked.

    Always the original characters: this is the left column, whose job is to show what was
    found. The safe version is `released_block_html()` below.
    """
    if not text:
        return "<div style='height:0.6rem'></div>"
    marks = []
    for item in items:
        start, end = item.get("start", 0), item.get("end", 0)
        if end <= start or end > len(text):
            continue
        style = PENDING_STYLE if item.get("action") == Action.ASK_HUMAN.value else \
            ACTION_STYLE.get(item.get("action", "KEEP"), ACTION_STYLE["KEEP"])
        title = f"{_pii_label(item['type'])} · {_action_label(item['action'])}" + (
            f" -> {item['replacement']}" if item.get("replacement") else "")
        marks.append((start, end, style, title, item["type"]))
    marks.sort(key=lambda m: m[0])

    out, cursor = [], 0
    for start, end, style, title, pii_type in marks:
        if start < cursor:
            continue
        out.append(html.escape(text[cursor:start]))
        out.append(_reveal_mark(text[start:end], pii_type, title))
        cursor = end
    out.append(html.escape(text[cursor:]))
    return f"<div style='margin:0 0 0.35rem 0;line-height:1.6'>{''.join(out)}</div>"


def released_block_html(block_text: str, items: list[dict],
                        applied: dict[str, dict]) -> str:
    """One block as it will actually be released, with every change marked.

    The safe version, not the original: the substituted value is what is shown for a
    REPLACE, a removed value leaves only a mark where it stood, and a kept value is
    highlighted in place. An item that is in neither list (`applied` has every decided
    item, one entry per id) is still waiting for a person and is absent from the release.

    Built by replaying the block in reading order rather than by highlighting the original
    text, because the released text is a different string: the offsets move.
    """
    out: list[str] = []
    cursor = 0
    for item in items:
        start, end = item.get("start", 0), item.get("end", 0)
        if end <= start or end > len(block_text) or start < cursor:
            continue
        out.append(html.escape(block_text[cursor:start]))
        decision = applied.get(item.get("id") or "")
        if decision is None:
            out.append(_gap(PENDING_STYLE,
                            f"{_pii_label(item['type'])} · {_t('Needs your review')}"))
        elif decision.get("action") == Action.REPLACE.value:
            out.append(_mark(decision.get("replacement") or "", ACTION_STYLE["REPLACE"],
                             f"{_pii_label(item['type'])} · {_action_label(Action.REPLACE.value)} "
                             f"({item['masked']})"))
        elif decision.get("action") == Action.REMOVE.value:
            out.append(_gap(ACTION_STYLE["REMOVE"],
                            f"{_pii_label(item['type'])} · {_action_label(Action.REMOVE.value)} "
                            f"({item['masked']})"))
        else:
            out.append(_mark(block_text[start:end], ACTION_STYLE["KEEP"],
                             f"{_pii_label(item['type'])} · {_action_label(Action.KEEP.value)} "
                             f"({item['masked']})"))
        cursor = end
    out.append(html.escape(block_text[cursor:]))
    return f"<div style='margin:0 0 0.35rem 0;line-height:1.6'>{''.join(out)}</div>"


def items_for_block(analysis, block_id: str) -> list[dict]:
    """Every item the model reported in this block, in reading order."""
    return [{"start": d.start, "end": d.end, "type": d.type, "action": d.action,
             "replacement": d.replacement, "id": d.meta.get("id"),
             "masked": d.meta.get("masked", "")}
            for d in sorted((d for d in analysis.detections
                             if d.block_id == block_id), key=lambda d: d.start)]


def badge(text: str, action: str) -> str:
    colour, background = ACTION_STYLE.get(action, ACTION_STYLE["KEEP"])
    label = _action_label(text) if text in ACTION_LABELS else _t(text)
    return (f"<span style='background:{background};color:{colour};padding:2px 8px;"
            f"border-radius:10px;font-size:0.78rem;font-weight:600'>{html.escape(label)}</span>")


_ARABIC_LETTER = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_RULE_ID = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b")           # VENDOR-PII-002


def _line_html(line: str) -> str:
    """One line of model text, with the direction that line actually has."""
    letters = _RULE_ID.sub("", line)
    rtl = len(_ARABIC_LETTER.findall(letters)) > len(_LATIN_LETTER.findall(letters))
    body = html.escape(line)
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    body = _RULE_ID.sub(lambda m: f"<bdi dir='ltr'>{m.group(0)}</bdi>", body)
    return (f"<div dir='{'rtl' if rtl else 'ltr'}' style='text-align:"
            f"{'right' if rtl else 'left'};line-height:1.9;margin:0 0 0.25rem 0'>{body}</div>")


def bidi_html(text: str) -> str:
    """Model text as HTML that reads in the right order, Arabic or English or both.

    Streamlit lays markdown out left-to-right. An Arabic answer with a Latin rule id inside
    it - "... (VENDOR-PII-002). ..." - is then reordered by the browser's bidirectional
    algorithm and the sentences come out jumbled. The fix is to give each line its real
    direction (right-to-left when it is mostly Arabic) and to keep every rule id as one
    left-to-right unit, so the punctuation around it stays where the sentence put it. The
    direction is decided per line, so an English line under an Arabic one is not flipped.
    The text is escaped first, so nothing the model writes can inject markup.
    """
    return "".join(_line_html(line) if line.strip() else "<div style='height:0.5rem'></div>"
                   for line in (text or "").splitlines())


def show_text(text: str) -> None:
    """Show model-written text with the right reading direction."""
    st.markdown(bidi_html(text), unsafe_allow_html=True)


# ------------------------------------------------------------------ page 1: upload


def page_upload() -> None:
    lib = default_library()
    state = engine.status()

    installed = bool(state["installed"])
    task_active = st.session_state.get("analysis_task") is not None
    has_analysis = st.session_state.get("analysis") is not None
    has_release = st.session_state.get("last") is not None
    active_step = 1 if task_active else 5 if has_release else 2 if has_analysis else 0
    receiver_keys = list(lib.receivers)
    default = st.session_state.get("selected_receiver", "EXTERNAL_VENDOR")
    if default not in receiver_keys:
        default = receiver_keys[0]
    selected_receiver_label = _t(lib.receiver_label(default))
    st.markdown(_lab_hero_markup(
        selected_receiver_label, len(lib.rules_for(default)), task_active),
        unsafe_allow_html=True)

    workflow_state = ("ANALYSIS IN PROGRESS" if task_active else
                      "PROTECTED COPY READY" if has_release else
                      "REVIEW READY" if has_analysis else "AWAITING SOURCE DOCUMENT")
    st.markdown(
        '<div class="workflow-shell"><div class="workflow-heading">'
        f'<span class="workflow-caption">{html.escape(_t("RELEASE SEQUENCE"))}</span>'
        f'<span class="workflow-state">{html.escape(_t(workflow_state))}</span></div>'
        f'{_workflow_markup(active_step, scanning=task_active)}</div>',
        unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{html.escape(_t("Choose a recipient"))}</div>',
                unsafe_allow_html=True)
    receiver = default
    for start in range(0, len(receiver_keys), 3):
        row_keys = receiver_keys[start:start + 3]
        receiver_columns = st.columns(len(row_keys))
        for column, key in zip(receiver_columns, row_keys):
            receiver_info = lib.receivers[key]
            receiver_rules = lib.rules_for(key)
            with column:
                selected = key == default
                with st.container(border=True):
                    st.markdown(_recipient_profile_markup(
                        key, receiver_info, receiver_rules, selected), unsafe_allow_html=True)
                    if st.button(_t("Selected" if selected else "Choose"),
                                 key=f"receiver-{key}", use_container_width=True,
                                 type="primary" if selected else "secondary"):
                        st.session_state["selected_receiver"] = key
                        st.rerun()
    receiver = st.session_state.get("selected_receiver", default)
    purpose = lib.receivers[receiver].purpose
    st.markdown(
        f'<div class="lab-footnote"><span class="signal-dot"></span>'
        f'<span>{html.escape(_t("SHARING PURPOSE"))}</span><span style="color:#c9c5bb">'
        f'{html.escape(_t(purpose))}</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{html.escape(_t("Add a document"))}</div>',
                unsafe_allow_html=True)
    upload_col, sample_col = st.columns([1.2, 1])
    with upload_col:
        upload = st.file_uploader(_t("Drop your document here or browse"),
                                  type=["pdf", "docx", "txt"],
                      help=_t("Supported formats: PDF, DOCX, TXT"))
    with sample_col:
        samples = sorted(SAMPLE_DIR.glob("*.docx")) + sorted(SAMPLE_DIR.glob("*.txt"))
        sample = st.selectbox(
            _t("Use a sample document"), ["none"] + [p.name for p in samples],
            format_func=lambda value: _t("Sample document") if value == "none" else value)
        if sample != "none":
            st.caption(_t("Synthetic sample selected · no external file is required."))

    # The purpose is stored with the audit record, so a release can be justified later.
    left, right = st.columns([1, 3])
    go = left.button(_t("Start privacy scan"), type="primary",
                     disabled=not installed, use_container_width=True)
    if state.get("provider") == "api":
        right.caption(_t("Document text may be processed by an external service."))
    else:
        right.caption(_t("Document content stays on this device."))

    if not installed:
        st.warning(_t("Analysis is unavailable. Ask your administrator to configure the privacy service."))
        return
    task = st.session_state.get("analysis_task")
    if go and task is None:
        document = None
        if upload is not None:
            try:
                document = readers.read_bytes(upload.name, upload.getvalue())
            except readers.UnsupportedDocument as exc:
                st.error(_t("This file type is not supported."))
                return
        elif sample != "none":
            path = SAMPLE_DIR / sample
            stored = readers.store_upload(path.name, path.read_bytes())
            try:
                document = readers.read(stored, path.name)
            finally:
                readers.discard(stored)
        else:
            st.warning(_t("Upload a document or choose a sample first."))
            return

        engine.reset_calls()
        started = time.perf_counter()
        future = _ANALYSIS_EXECUTOR.submit(
            analyze, document, receiver, purpose)
        task = {"future": future, "started": started}
        st.session_state["analysis_task"] = task
        st.rerun()

    if task is None:
        return

    future: Future = task["future"]
    started = task["started"]
    with st.status(_t("Checking this document against recipient safeguards…"),
                   expanded=True) as status_box:
        progress_line = status_box.empty()
        progress_line.markdown(
            f'<div class="scan-live"><span class="scan-live-beam"></span>'
            f'<span>{html.escape(_t("Scanning in progress"))}</span></div>',
            unsafe_allow_html=True)
        while not future.done():
            time.sleep(0.25)
        try:
            analysis = future.result()
        except engine.ModelNotInstalled:
            status_box.update(label=_t("Analysis is unavailable."), state="error")
            st.error(_t("Analysis is unavailable. Ask your administrator to configure the privacy service."))
            st.session_state.pop("analysis_task", None)
            return
        except Exception:                                           # noqa: BLE001
            status_box.update(label=_t("The review could not be completed. No document was changed."), state="error")
            st.error(_t("The review could not be completed. No document was changed."))
            st.session_state.pop("analysis_task", None)
            return
        elapsed = time.perf_counter() - started
        status_box.update(
            label=f"{_t('Scan complete')} · {len(analysis.detections)} {_t('items found')}",
            state="complete", expanded=False)

    st.session_state.pop("analysis_task", None)
    st.session_state["analysis"] = analysis
    st.session_state["decisions"] = {}
    st.session_state["page"] = "Review"
    st.session_state["analysis_elapsed"] = elapsed
    st.success(f"{len(analysis.detections)} {_t('items checked. Opening privacy review.')}")
    st.rerun()


# ------------------------------------------------------------------ page 2: review


def page_review() -> None:
    analysis = st.session_state.get("analysis")
    if analysis is None:
        _hero("PRIVACY REVIEW", "Nothing to review yet.",
              "Run an analysis to compare the original with a receiver-safe release.")
        if st.button(_t("Go to upload"), type="primary"):
            st.session_state["page"] = "Upload"
            st.rerun()
        return

    lib = default_library()
    decisions: dict[str, str] = st.session_state.setdefault("decisions", {})
    queue = build_queue(analysis)
    state = review_state(analysis, decisions)
    summary = analysis.summary()

    _hero("REVIEW WORKSPACE", analysis.document.filename,
          f"{_t(lib.receiver_label(analysis.receiver))} · {_t(analysis.purpose)}")

    columns = st.columns(5)
    columns[0].metric(_t("Detected"), summary["items"])
    columns[1].metric(_t("Replaced"), summary["by_action"]["REPLACE"])
    columns[2].metric(_t("Removed"), summary["by_action"]["REMOVE"])
    columns[3].metric(_t("Retained"), summary["by_action"]["KEEP"])
    columns[4].metric(_t("Human review"), summary["needs_review"])

    if summary["needs_review"]:
        warning = _t("{count} item(s) need your review. They stay out of the shared copy until decided ({answered}/{total} reviewed).")
        st.warning(warning.format(count=summary["needs_review"],
                                  answered=state["answered"], total=state["total"]))

    if analysis.notes:
        with st.expander(f"{_t('Safeguard notes')} · {len(analysis.notes)}"):
            for note in analysis.notes:
                st.warning(_guardrail_note(note), icon=":material/shield:")

    total = summary["items"]
    protected = summary["by_action"]["REPLACE"] + summary["by_action"]["REMOVE"]
    coverage = (protected / total) if total else 0
    st.markdown(f'<div class="section-label">{html.escape(_t("Before protection"))}</div>',
                unsafe_allow_html=True)
    exp_left, exp_right = st.columns([1, 2])
    exp_left.metric(_t("Protected items"), f"{protected} / {total}",
                    help=_t("Count of items protected by recipient policy."))
    exp_right.progress(coverage, text=f"{coverage:.0%} {_t('of detected items protected')}")

    tab_compare, tab_items = st.tabs([_t("Before protection"), _t("Review findings")])

    with tab_compare:
        # The right column shows the released text, so it is built with the same call the
        # "Generate release" button makes: the model's proposal plus anything you have
        # answered so far, with every undecided item left OUT of the release.
        preview = apply_review(analysis, decisions)
        applied = {item["id"]: item for item in preview["applied"]}
        pending_by_block: dict[str, int] = {}
        for item in preview["pending"]:
            pending_by_block[item["block_id"]] = pending_by_block.get(item["block_id"], 0) + 1

        left, right = st.columns(2)
        left.markdown(f"#### {_t('Original document')}")
        right.markdown(f"#### {_t('Protected preview')}")
        st.caption(_t("Click a redacted value to reveal it temporarily."))
        for block in analysis.document.blocks:
            if not block.text.strip():
                continue
            items = items_for_block(analysis, block.id)
            with left:
                st.markdown(highlight_block(block.text, items), unsafe_allow_html=True)
            with right:
                waiting = pending_by_block.get(block.id, 0)
                if waiting:
                    st.markdown(
                        f"<div style='margin:0 0 0.35rem 0;color:{PENDING_STYLE[0]};"
                        f"background:{PENDING_STYLE[1]};padding:2px 6px;border-radius:3px'>"
                        f"{waiting} {_t('items need your review')}</div>",
                        unsafe_allow_html=True)
                st.markdown(released_block_html(block.text, items, applied),
                            unsafe_allow_html=True)
        legend = " &nbsp; ".join(badge(action, action) for action in ACTION_STYLE)
        st.markdown(f"<div style='margin-top:1rem'>{legend}</div>", unsafe_allow_html=True)
        st.caption(_t("Values awaiting your decision are excluded from the shared copy."))

    with tab_items:
        filter_cols = st.columns(3)
        type_options = [_t("All types")] + sorted({d.type for d in analysis.detections})
        action_options = [_t("All actions")] + sorted({d.action for d in analysis.detections})
        selected_type = filter_cols[0].selectbox(
            _t("PII type"), type_options, key="review_type",
            format_func=lambda value: value if value == _t("All types") else _pii_label(value))
        selected_action = filter_cols[1].selectbox(
            _t("Action"), action_options, key="review_action",
            format_func=lambda value: value if value == _t("All actions") else _action_label(value))
        confidence_floor = filter_cols[2].slider(_t("Minimum confidence"), 0.0, 1.0, 0.0,
                                                  0.05, key="review_confidence")
        filtered_decisions = [
            d for d in analysis.detections
            if d.action != Action.ASK_HUMAN.value
            and (selected_type == _t("All types") or d.type == selected_type)
            and (selected_action == _t("All actions") or d.action == selected_action)
            and d.confidence >= confidence_floor]
        if filtered_decisions:
            st.markdown(f"#### {_t('Findings')}")
            for detection in filtered_decisions:
                color = CHART_COLORS.get(detection.action, CHART_COLORS["OTHER"])
                with st.container(border=True):
                    card_left, card_right = st.columns([3, 1])
                    with card_left:
                        st.markdown(
                            f"**{html.escape(_pii_label(detection.type))}**　<span class='action-chip'>"
                            f"{html.escape(_action_label(detection.action))}</span>　"
                            f"<span style='color:{color}'>●</span> "
                            f"`{html.escape(detection.meta.get('masked', ''))}`",
                            unsafe_allow_html=True)
                        st.caption(f"{_t('Source policy')} · {detection.rule_id or _t('No valid rule')}"
                                   + (f" · {detection.reason}" if detection.reason else ""))
                        if detection.action == Action.REPLACE.value and detection.replacement:
                            st.caption(f"{_t('Validated safe substitute')} · `{detection.replacement}`")
                    with card_right:
                        st.metric(_t("Confidence"), f"{detection.confidence:.0%}")
                        st.progress(max(0.0, min(1.0, detection.confidence)))
        elif analysis.detections:
            st.info(_t("No findings match your filters."))

        filtered_queue = [
            item for item in queue
              if (selected_type == _t("All types") or item.type == selected_type)
              and (selected_action == _t("All actions")
                 or selected_action == Action.ASK_HUMAN.value)
            and item.confidence >= confidence_floor]
        if not queue:
            st.success(_t("Every detected item has a policy-backed decision."))
        elif filtered_queue:
            st.markdown(f"**{_t('Waiting for your decision')}**")
        else:
            st.info(_t("No open review items match these filters."))
        for item in filtered_queue:
            decision = decisions.get(item.id)
            with st.container(border=True):
                head, meta = st.columns([3, 2])
                with head:
                    st.markdown(f"**{_pii_label(item.type)}** &nbsp; {badge('Needs your review', 'ASK_HUMAN')}",
                                unsafe_allow_html=True)
                    st.markdown(f"`{item.original_text}` → **{_action_label(item.proposed_action)}**"
                                + (f" → `{item.proposed_replacement}`"
                                   if item.proposed_replacement else ""))
                    st.markdown(f"**{_t('Cited policy rule')}:** "
                                f"`{item.rule_id or 'none'}`  \n{item.rule_text}")
                    st.markdown(f"**{_t('Why flagged')}:** {item.reason}")
                with meta:
                    st.metric(_t("Confidence"), f"{item.confidence:.2f}")
                    st.caption(f"{_t('Reason for review')}: {_review_reason(item.ask_reason)}")
                    if decision:
                        st.success(f"{_t('Your decision')}: {_action_label(decision)}")
                buttons = st.columns(4)
                for index, action in enumerate(
                        (Action.KEEP.value, Action.REPLACE.value, Action.REMOVE.value)):
                    if buttons[index].button(_action_label(action), key=f"decision-{action.lower()}-{item.id}",
                                             width="stretch"):
                        decisions[item.id] = action
                        st.rerun()
                if buttons[3].button(
                    _action_label("REJECT_SUGGESTION"), key=f"reject-{item.id}",
                    width="stretch",
                    help=_t("Apply nothing from this suggestion. It remains excluded until you decide.")):
                    decisions[item.id] = "REJECT_SUGGESTION"
                    st.rerun()

        if queue:
            st.divider()
            if st.toggle(_t("Accept suggested decisions for remaining items"),
                         value=False,
                         help=_t("Each choice is recorded as a reviewer decision.")):
                for item in queue:
                    decisions.setdefault(item.id, item.proposed_action)
                st.rerun()

    st.divider()
    left, right = st.columns([1, 3])
    if left.button(_t("Generate protected copy"), type="primary"):
        applied = apply_review(analysis, decisions)
        out_dir = config.TMP_DIR / "releases"
        written = write_release(analysis.document, out_dir,
                                Path(analysis.document.filename).stem, applied["changed"])
        with st.spinner(_t("Preparing the protected copy…")):
            summary_text = write_summary(analysis, applied)
        run_id = audit.record_run(analysis, applied, ai_summary=summary_text)
        st.session_state["last"] = {"applied": applied, "written": written,
                                    "run_id": run_id, "summary": summary_text}

    last = st.session_state.get("last")
    if last:
        applied, written = last["applied"], last["written"]
        st.caption(f"{_t('Document format')}: {written['format']} · "
                   f"{_t('Layout preserved') if written['layout_preserved'] else _t('Layout may differ')}"
                   + (f" ({written.get('note', '')})" if written.get("note") else ""))
        if applied["incomplete"]:
            st.warning(_t("Some items still need your decision and are excluded from the shared copy."))
        info = st.columns(3)
        info[0].success(f"{len(applied['applied'])} {_t('changes applied')}")
        info[1].success(_t("Your protected copy is ready."))
        info[2].download_button(_t("Download protected copy"),
                                data=Path(written["path"]).read_bytes(),
                                file_name=Path(written["path"]).name)
        if last.get("summary"):
            st.info(f"**{_t('Release note')}:** {last['summary']}")
        with st.expander(_t("Open protected text")):
            st.code(safe_text(analysis.document, applied["changed"]), language=None)
        with st.expander(_t("Release reference")):
            st.code(last["run_id"], language=None)


# ------------------------------------------------------------------- page 3: audit


def page_audit() -> None:
    _hero("SECURITY ANALYTICS", "Audit intelligence",
          "Recorded sharing activity and decisions.")
    runs = audit.runs(limit=1000)
    rows = audit.recent(limit=5000)
    if not runs and not rows:
        st.info(_t("No recorded activity yet."))
        return

    run_frame = pd.DataFrame(runs)
    decision_frame = pd.DataFrame(rows)
    for frame in (run_frame, decision_frame):
        if not frame.empty and "created_at" in frame:
            frame["event_time"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)

    filter_cols = st.columns(4)
    library = default_library()
    receiver_labels = {key: value.label for key, value in library.receivers.items()}
    receivers = sorted(set(
        (run_frame["receiver"].dropna().astype(str).tolist() if not run_frame.empty else [])
        + (decision_frame["receiver"].dropna().astype(str).tolist()
           if not decision_frame.empty else [])))
    receiver_filter = filter_cols[0].selectbox(
        _t("Recipient"), ["All receivers"] + receivers, key="audit_receiver",
        format_func=lambda value: _t("All recipients") if value == "All receivers"
        else _t(receiver_labels.get(value, value)))
    documents = sorted(set(
        (run_frame["document"].dropna().astype(str).tolist() if not run_frame.empty else [])
        + (decision_frame["document"].dropna().astype(str).tolist()
           if not decision_frame.empty else [])))
    document_labels = {
        label: filename for index, filename in enumerate(documents, start=1)
        for label in [f"Document {index:02d}"]
    }
    document_filter = filter_cols[1].selectbox(
        _t("Shared document"), ["All documents"] + list(document_labels), key="audit_document",
        format_func=lambda value: _t("All documents") if value == "All documents"
        else f"{_t('Shared document')} {value.split()[-1]}")
    actions = (sorted(decision_frame["action"].dropna().astype(str).unique().tolist())
               if not decision_frame.empty else [])
    action_filter = filter_cols[2].selectbox(
        _t("Action"), ["All actions"] + actions, key="audit_action",
        format_func=lambda value: _t("All decisions") if value == "All actions"
        else _action_label(value))

    date_values = []
    for frame in (run_frame, decision_frame):
        if not frame.empty and "event_time" in frame:
            date_values.extend(frame["event_time"].dropna().dt.date.tolist())
    if date_values:
        date_range = filter_cols[3].date_input(
            _t("Filter by date"), value=(min(date_values), max(date_values)),
            min_value=min(date_values), max_value=max(date_values), key="audit_dates")
        if isinstance(date_range, tuple) and len(date_range) == 2:
            date_start, date_end = date_range
        elif isinstance(date_range, date):
            date_start = date_end = date_range
        else:
            date_start, date_end = min(date_values), max(date_values)
    else:
        date_start = date_end = None

    filtered_runs = run_frame.copy()
    filtered_decisions = decision_frame.copy()
    for frame in (filtered_runs, filtered_decisions):
        if frame.empty:
            continue
        if receiver_filter != "All receivers":
            frame.drop(frame.index[frame["receiver"] != receiver_filter], inplace=True)
        if document_filter != "All documents":
            frame.drop(frame.index[
                frame["document"] != document_labels[document_filter]], inplace=True)
        if date_start and date_end and "event_time" in frame:
            event_dates = frame["event_time"].dt.date
            frame.drop(frame.index[(event_dates < date_start) | (event_dates > date_end)],
                       inplace=True)
    if not filtered_runs.empty and not filtered_decisions.empty:
        selected_run_ids = set(filtered_runs["run_id"].astype(str))
        filtered_decisions = filtered_decisions[
            filtered_decisions["run_id"].astype(str).isin(selected_run_ids)]
    if action_filter != "All actions" and not filtered_decisions.empty:
        filtered_decisions = filtered_decisions[
            filtered_decisions["action"] == action_filter]
    if not filtered_runs.empty:
        matching_run_ids = (set(filtered_decisions["run_id"].astype(str))
                            if not filtered_decisions.empty else set())
        filtered_runs = filtered_runs[
            filtered_runs["run_id"].astype(str).isin(matching_run_ids)]

    decisions_count = len(filtered_decisions)
    replacements = int((filtered_decisions["action"] == Action.REPLACE.value).sum()) \
        if decisions_count else 0
    removals = int((filtered_decisions["action"] == Action.REMOVE.value).sum()) \
        if decisions_count else 0
    review_count = int((filtered_decisions["action"] == Action.ASK_HUMAN.value).sum()) \
        if decisions_count else 0
    kpis = st.columns(5)
    kpis[0].metric(_t("Runs"), len(filtered_runs))
    kpis[1].metric(_t("Findings"), decisions_count)
    kpis[2].metric(_action_label(Action.REPLACE.value), replacements)
    kpis[3].metric(_action_label(Action.REMOVE.value), removals)
    kpis[4].metric(_t("Human reviews"), review_count)

    if not filtered_runs.empty:
        st.markdown(f"#### {_t('Decision timeline')}")
        timeline_entries = []
        ordered_runs = (filtered_runs.sort_values("event_time", ascending=False)
                        if "event_time" in filtered_runs else filtered_runs)
        receiver_labels = {key: value.label
                           for key, value in default_library().receivers.items()}
        for _, run in ordered_runs.head(8).iterrows():
            event_time = run.get("event_time")
            when = (event_time.strftime("%b %d, %Y · %H:%M UTC")
                    if pd.notna(event_time) else "Timestamp not recorded")
            document_name = html.escape(str(run.get("document") or "Document not recorded"))
            receiver_key = str(run.get("receiver") or "")
            run_id = html.escape(str(run.get("run_id") or "Run id not recorded"))
            raw_count = run.get("items")
            item_count = "not recorded" if pd.isna(raw_count) else str(raw_count)
            receiver_name = _t(receiver_labels.get(receiver_key, receiver_key or "Receiver not recorded"))
            timeline_entries.append(
                '<article class="timeline-entry">'
                f'<div><div class="timeline-date">{html.escape(_t(when))}</div>'
                f'<div class="timeline-detail">{html.escape(receiver_name)} · {html.escape(item_count)} {_t("items found")}</div></div>'
                f'<div><div class="timeline-document">{document_name}</div>'
                f'<div class="timeline-detail">{html.escape(_t("Recorded sharing activity"))}</div></div>'
                f'<div class="timeline-run">{html.escape(_t("Release reference"))} {run_id}</div></article>')
        st.markdown('<div class="decision-timeline">' + "".join(timeline_entries)
                    + '</div>', unsafe_allow_html=True)

    if filtered_decisions.empty:
        st.info(_t("No findings match your filters."))
    else:
        action_counts = (filtered_decisions["action"].value_counts()
                         .rename_axis("action").reset_index(name="count"))
        left, right = st.columns(2)
        with left:
            st.markdown(f"#### {_t('Decision distribution')}")
            action_counts["label"] = action_counts["action"].map(_action_label)
            localized_colors = {_action_label(key): color for key, color in CHART_COLORS.items()}
            donut = px.pie(action_counts, names="label", values="count", hole=.68,
                           color="label", color_discrete_map=localized_colors)
            donut.update_traces(textposition="inside", textinfo="percent+label",
                                marker=dict(line=dict(color="#0b1220", width=3)))
            _chart(donut, "audit-action-donut", 310)
        with right:
            st.markdown(f"#### {_t('Protected data categories')}")
            by_type = (filtered_decisions.groupby(["item_type", "action"], as_index=False)
                       .size().rename(columns={"size": "count"}))
            by_type["type_label"] = by_type["item_type"].map(_pii_label)
            by_type["action_label"] = by_type["action"].map(_action_label)
            bars = px.bar(by_type, x="type_label", y="count", color="action_label",
                          barmode="stack", color_discrete_map=localized_colors)
            bars.update_layout(xaxis_title="", yaxis_title=_t("Findings"))
            bars.update_xaxes(tickangle=-25)
            _chart(bars, "audit-pii-categories", 310)

        left, right = st.columns(2)
        with right:
            st.markdown(f"#### {_t('Activity by receiver')}")
            receiver_activity = (filtered_decisions.groupby(
                ["receiver", "action"], as_index=False).size()
                .rename(columns={"size": "decisions"}))
            receiver_activity["receiver_label"] = receiver_activity["receiver"].map(
                lambda value: _t(receiver_labels.get(value, value)))
            receiver_activity["action_label"] = receiver_activity["action"].map(_action_label)
            by_receiver = px.bar(receiver_activity, x="receiver_label", y="decisions",
                                 color="action_label", barmode="stack",
                                 color_discrete_map=localized_colors)
            by_receiver.update_layout(xaxis_title="", yaxis_title=_t("Findings"))
            by_receiver.update_xaxes(tickangle=-15)
            _chart(by_receiver, "audit-receiver-activity", 290)

    st.markdown(f"#### {_t('Release activity')}")
    if not filtered_runs.empty:
        run_summaries = []
        for _, run in filtered_runs.iterrows():
            run_summaries.append({
                _t("Release reference"): run.get("run_id"),
                _t("Time"): run.get("created_at"),
                _t("Document"): next((label for label, name in document_labels.items()
                                       if name == run.get("document")), _t("Shared document")),
                _t("Recipient"): _t(receiver_labels.get(run.get("receiver"), run.get("receiver", ""))),
                _t("Findings"): run.get("items"),
            })
        st.dataframe(pd.DataFrame(run_summaries), use_container_width=True, hide_index=True)

    if not filtered_decisions.empty:
        filtered_decisions = filtered_decisions.copy()
        filtered_decisions["document_label"] = filtered_decisions["document"].map(
            {filename: label for label, filename in document_labels.items()})
        safe_columns = [("created_at", _t("Time")), ("document_label", _t("Document")),
                ("receiver", _t("Recipient")), ("item_type", _t("PII type")),
                ("value_masked", _t("Masked value")), ("action", _t("Action")),
                ("rule_id", _t("Source policy")),
                ("confidence", _t("Confidence")), ("decided_by", _t("Decision source"))]
        shown = filtered_decisions[[column for column, _ in safe_columns
                                    if column in filtered_decisions.columns]].rename(
                                        columns=dict(safe_columns))
        if "Masked value" in shown:
            shown["Masked value"] = shown["Masked value"].map(_mask_audit_value)
        if "Action" in shown:
            shown["Action"] = shown["Action"].map(_action_label)
        if "PII type" in shown:
            shown["PII type"] = shown["PII type"].map(_pii_label)
        if "Recipient" in shown:
            shown["Recipient"] = shown["Recipient"].map(
                lambda value: _t(receiver_labels.get(value, value)))
        if "Decision source" in shown:
            shown["Decision source"] = shown["Decision source"].map(
                lambda value: _t({"system": "Automatic policy decision",
                                  "unresolved": "Needs your review"}.get(value, value)))
        with st.expander(f"{_t('Audit trail')} · {len(shown)}"):
            st.caption(_t("Values are masked. The table omits raw values, explanations, and replacement text."))
            st.dataframe(shown, use_container_width=True, hide_index=True)
            csv_data = shown.to_csv(index=False).encode("utf-8")
            st.download_button(_t("Download audit file"), data=csv_data,
                               file_name="sitr-audit-masked.csv", mime="text/csv")
            st.markdown(f"**{_t('Policy and decision trace')}**")
            st.caption(_t("This trace shows the recorded action, source policy, and reviewer."))
            for _, row in filtered_decisions.iterrows():
                st.markdown(
                    f"- `{_mask_audit_value(row.get('value_masked', ''))}` · "
                    f"{_pii_label(row.get('item_type', ''))} "
                    f"→ **{_action_label(row.get('action', ''))}** · {_t('Source policy')} "
                    f"`{row.get('rule_id') or 'none'}` · {row.get('decided_by') or _t('system')}")


# ----------------------------------------------------------------- page 4: policy

POLICY_QUICK_QUESTIONS = [
    "May an external vendor see a phone number?",
    "What may HR keep in an employee record?",
    "هل يمكن مشاركة رقم الهوية مع مورد خارجي؟",
    "What does the analyst receive?",
    "What happens if no rule covers an item?",
]


def page_policy() -> None:
    """Ask the policy a question (Chroma + the local model), and edit the rules.

    Two things about this page are deliberate:

    * the **question answering** is a reading aid. It retrieves rules and cites them;
      it never sees a document, and nothing it says is applied to one. If the model is
      not installed the retrieved rules are still shown, unsummarised.
    * the **editing** path is the only way the app writes `rules.json`. It validates the
      whole library, writes atomically, bumps the version and records who changed what -
      see `app/policy/policy_editor.py`.
    """
    lib = default_library()
    editor = PolicyEditor()

    try:
        status = get_index().status()
        index_error = status.get("error", "")
    except PolicyIndexError as exc:
        status, index_error = {}, str(exc)

    _hero("POLICY INTELLIGENCE", "One policy. Every release.",
          "Explore receiver coverage, ask grounded questions, and audit policy changes.")
    columns = st.columns(2)
    columns[0].metric(_t("Active rules"), len(lib.rules))
    columns[1].metric(_t("Recipients"), len(lib.receivers))
    coverage_entries = list(lib.receivers.items())
    for start in range(0, len(coverage_entries), 3):
        coverage = st.columns(min(3, len(coverage_entries) - start))
        for column, (key, receiver_info) in zip(coverage, coverage_entries[start:start + 3]):
            rule_count = len(lib.rules_for(key))
            with column:
                with st.container(border=True):
                    st.markdown(f"**{html.escape(_t(receiver_info.label))}**")
                    st.metric(_t("Sharing rules"), rule_count)
                    st.caption(_t(receiver_info.description))

    if index_error or not status.get("installed"):
        st.info(_t("Policy search is unavailable right now. The sharing rules below remain available."))
        with st.expander(_t("Technical details")):
            if status.get("model"):
                st.caption(f"{_t('Embedding model')}: {status['model']}")
            if index_error:
                st.code(index_error)
            elif status.get("setup_command"):
                st.code(status["setup_command"], language="bash")

    tab_ask, tab_rules, tab_history, tab_document = st.tabs(
        [_t("Search policies"), _t("Browse policies"), _t("Policy history"), _t("Document assistant")])
    with tab_ask:
        policy_ask_tab(lib, status)
    with tab_rules:
        policy_rules_tab(lib, editor)
    with tab_history:
        policy_history_tab(editor)
    with tab_document:
        policy_document_tab()


def policy_ask_tab(lib, status: dict) -> None:
    st.markdown(f"### ◈  {_t('Search policies')}")
    st.caption(_t("Ask in Arabic or English. Answers are based on the matching sharing rules."))

    left, right = st.columns([2, 1])
    with left:
        question = st.text_input(_t("Your question"),
                     placeholder=_t("May an external vendor see a phone number?"),
                                 key="policy_question")
        quick = st.selectbox(_t("Select a quick question"), ["none"] + POLICY_QUICK_QUESTIONS,
                     format_func=lambda value: _t("No quick question")
                     if value == "none" else _t(value))
        if quick != "none":
            question = quick
    with right:
        receiver_keys = ["any"] + list(lib.receivers)
        receiver = st.selectbox(
            _t("Recipient filter"), receiver_keys, key="policy_receiver",
            format_func=lambda key: _t("Any recipient") if key == "any"
            else _t(lib.receivers[key].label))
        type_filter = st.selectbox(_t("Type filter"), ["any"] + [t.value for t in PiiType],
                                   key="policy_type",
                                   format_func=lambda value: _t("Any type")
                                   if value == "any" else _pii_label(value))
        use_model = st.toggle(
            _t("Write a concise answer"), value=True, key="policy_use_model",
            disabled=not status.get("installed"),
            help=_t("Turn this off to show matching rules without a written summary."))

    if st.button(_t("Search policies"), type="primary", key="policy_ask"):
        with st.spinner(_t("Searching the policy...")):
            try:
                result = policy_qa.ask_policy(
                    question, receiver=None if receiver == "any" else receiver,
                    pii_type=None if type_filter == "any" else type_filter,
                    use_model=use_model)
            except PolicyIndexError:
                st.error(_t("Policy search is unavailable right now. The sharing rules below remain available."))
                return
        st.session_state["policy_answer"] = result

    result = st.session_state.get("policy_answer")
    if not result:
        return
    st.divider()
    for note in result.get("notes", []):
        st.info(_policy_note(note))

    if result.get("answer"):
        st.markdown(f"**{_t('Policy answer')}**")
        show_text(result["answer"])
        if result.get("cited"):
            st.caption(_t("Verified citations: ") + ", ".join(f"`{rule_id}`"
                                                          for rule_id in result["cited"]))
        for suspect in result.get("suspect_citations", []):
            st.warning(f"{_t('A citation was adjusted to the matching policy rule')}: "
                       f"`{suspect['meant']}` ({_action_label(suspect['action'])}).")
        if result.get("invented_citations"):
            st.error(_t("An unsupported rule reference was omitted. Use the matching rules below."))
        st.caption(_t("Policy guidance supports your review; it does not make the sharing decision."))
        if result.get("model"):
            with st.expander(_t("Technical details")):
                st.caption(f"{_t('Model name')}: {result['model']} · "
                           f"{_t('Inference time')}: {result.get('seconds', 0)}s")
    elif result.get("hits"):
        st.markdown(f"**{_t('Rules that support this answer')}**")

    shown = {"receiver": result.get("receiver_filter") or "any",
             "type": result.get("type_filter") or "any"}
    st.caption(f"{_t('Recipient')}: {_t(lib.receiver_label(shown['receiver'])) if shown['receiver'] in lib.receivers else _t('Any recipient')} · "
               f"{_t('Type filter')}: {_pii_label(shown['type']) if shown['type'] != 'any' else _t('Any type')} · "
               f"{len(result.get('hits', []))} {_t('matching rules')}")

    for hit in result.get("hits", []):
        with st.container(border=True):
            head, relevance = st.columns([4, 1])
            head.markdown(
                f"<span class='rule-id'>{html.escape(hit['rule_id'])}</span>　"
                f"<span class='action-chip'>{html.escape(_action_label(hit['action']))}</span>　"
                f"**{html.escape(_pii_label(hit['pii_type']))}** · "
                f"{html.escape(_t(lib.receiver_label(hit['receiver'])))}",
                unsafe_allow_html=True)
            relevance.caption(_t("Matching rule"))
            st.markdown(hit["reason"])
            if hit.get("citation"):
                st.caption(f"{_t('Policy source')} · {hit['citation']}")

    if result.get("hits"):
        with st.expander(_t("Rules that support this answer")):
            st.code(result.get("grounded") or policy_qa.grounded_answer(
                result, result["hits"], lib), language=None)
            st.caption(_t("This answer is guidance for your review; it does not change a decision or document."))
        with st.expander(_t("Use these rules during review")):
            st.caption(_t("A policy answer is context for a reviewer, never an instruction to the release pipeline."))
            st.code(result.get("grounded") or result.get("answer") or "", language=None)


def policy_rules_tab(lib, editor: PolicyEditor) -> None:
    receiver_keys = ["all"] + list(lib.receivers)
    left, right = st.columns([1, 3])
    with left:
        receiver = st.selectbox(
            _t("Recipient"), receiver_keys, key="policy_edit_receiver",
            format_func=lambda key: _t("All recipients") if key == "all"
            else _t(lib.receivers[key].label))
    rules = editor.rules(None if receiver == "all" else receiver)
    with right:
        st.caption(f"{len(rules)} {_t('sharing rules')}. {_t('Edit the sharing action, explanation, or source.')}")

    action_counts = {a.value: 0 for a in Action}
    for rule in rules:
        action_counts[rule.action] = action_counts.get(rule.action, 0) + 1
    st.caption(f"{_t('Actions in this selection')}: " + ", ".join(
        f"{_action_label(value)} {count}" for value, count in action_counts.items()))

    if rules:
        st.markdown(f"### {_t('Sharing rules')}")
        groups = {key: [rule for rule in rules if rule.receiver == key]
                  for key in (lib.receivers if receiver == "all" else [receiver])}
        group_columns = st.columns(min(3, max(1, len(groups))))
        for index, (key, group_rules) in enumerate(groups.items()):
            with group_columns[index % len(group_columns)]:
                st.markdown(f"#### {_t(lib.receiver_label(key))}")
                for rule in group_rules:
                    color = CHART_COLORS.get(rule.action, CHART_COLORS["OTHER"])
                    st.markdown(
                        f"<div class='rule-card'><div class='rule-id'>"
                        f"{html.escape(rule.rule_id)}</div><div style='margin:.4rem 0'>"
                        f"<span class='action-chip' style='color:{color}'>"
                        f"{html.escape(_action_label(rule.action))}</span>　"
                        f"<span class='muted'>{html.escape(_pii_label(rule.pii_type))}</span></div>"
                        f"<div class='rule-reason'>{html.escape(rule.reason)}</div>"
                        f"<div class='muted'>{html.escape(rule.citation or _t('No source citation'))}</div>"
                        f"</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader(_t("Change a rule"))
    rule_ids = [rule.rule_id for rule in rules]
    if not rule_ids:
        st.info(_t("This recipient has no sharing rules. Sensitive items are not automatically approved."))
    else:
        chosen_id = st.selectbox(_t("Rule"), rule_ids, key="policy_edit_rule")
        chosen = editor.rule(chosen_id)
        with st.form("policy-edit-form", clear_on_submit=False):
            columns = st.columns([1, 2])
            with columns[0]:
                st.text_input(_t("Rule id (fixed)"), value=chosen.rule_id, disabled=True)
                st.text_input(_t("Type (fixed)"), value=_pii_label(chosen.pii_type), disabled=True)
                st.text_input(_t("Recipient (fixed)"),
                              value=_t(lib.receiver_label(chosen.receiver)), disabled=True)
            with columns[1]:
                actions = [action.value for action in Action]
                new_action = st.selectbox(
                    _t("Sharing action"), actions, index=actions.index(chosen.action),
                    format_func=_action_label,
                    help=_t("Choose whether the data is shared as-is, replaced, removed, or held for review."))
                new_reason = st.text_area(_t("Sharing explanation"), value=chosen.reason, height=90,
                                          help=_t("This explanation appears to reviewers and in the audit record."))
                new_citation = st.text_input(_t("Policy source"), value=chosen.citation)
            editor_name = st.text_input(_t("Your name (recorded in the change history)"),
                                        value="reviewer")
            confirmed = st.checkbox(
                _t("I understand this changes the policy for every future analysis."))
            submitted = st.form_submit_button(_t("Save the new policy"), type="primary",
                                              disabled=not confirmed)
        if submitted:
            try:
                report = editor.set_rule(chosen_id, action=new_action, reason=new_reason,
                                         citation=new_citation, editor=editor_name)
            except PolicyEditError as exc:
                st.error(_t("The policy change was not saved."))
                with st.expander(_t("Technical details")):
                    st.code(str(exc))
            else:
                if not report.get("changed"):
                    st.info(_t("Nothing to update; this rule already has those settings."))
                else:
                    st.success(f"{chosen_id} · {_t('Rule saved.')}")
                    for change in report.get("changes", []):
                        st.markdown(f"- `{change}`")
                    if report.get("index", {}).get("orphaned"):
                        st.caption(f"The index moved to a new collection for this version; "
                                   f"{report['index']['orphaned']} older collection(s) are "
                                   f"left behind. Remove them with "
                                   f"`python -m scripts.setup_model --rebuild-index`.")
                    if report.get("audit_error"):
                        st.warning(_t("The rule was saved, but its audit record could not be written."))
                    if report.get("index_error"):
                        st.warning(_t("The rule was saved. Policy search will refresh when available."))
                    st.caption(_t("The next document scan will use this rule."))

        with st.expander(_t("Delete this rule")):
            st.caption(_t("Removing this rule means sensitive items will wait for review."))
            typed = st.text_input(f"{_t('Type to confirm')}: {chosen_id}", key="policy_delete_confirm")
            if st.button(_t("Delete the rule"), key="policy_delete",
                         disabled=typed.strip() != chosen_id):
                try:
                    report = editor.delete_rule(chosen_id, editor="reviewer")
                except PolicyEditError as exc:
                    st.error(_t("The rule could not be deleted."))
                    with st.expander(_t("Technical details")):
                        st.code(str(exc))
                else:
                    st.success(f"{chosen_id} · {_t('Rule deleted.')}")

    with st.expander(_t("Add a sharing rule")):
        st.caption(_t("Each recipient can have one rule for each data type."))
        with st.form("policy-add-form", clear_on_submit=True):
            new_id = st.text_input(_t("New rule id"), placeholder="VENDOR-PII-013")
            columns = st.columns(3)
            new_receiver = columns[0].selectbox(
                _t("Recipient"), list(lib.receivers), format_func=lambda key: _t(lib.receiver_label(key)))
            new_type = columns[1].selectbox(
                _t("PII type"), [t.value for t in PiiType], format_func=_pii_label)
            new_action = columns[2].selectbox(
                _t("Sharing action"), [a.value for a in Action], format_func=_action_label)
            new_reason = st.text_area(_t("Sharing explanation"), height=80)
            new_citation = st.text_input(_t("Policy source"))
            confirm_add = st.checkbox(_t("I understand this adds a rule to the live policy."))
            added = st.form_submit_button(_t("Add the rule"), type="primary",
                                          disabled=not confirm_add)
        if added:
            try:
                report = editor.add_rule(rule_id=new_id, receiver=new_receiver,
                                         pii_type=new_type, action=new_action,
                                         reason=new_reason, citation=new_citation,
                                         editor="reviewer")
            except PolicyEditError as exc:
                st.error(_t("The new rule could not be saved."))
                with st.expander(_t("Technical details")):
                    st.code(str(exc))
            else:
                st.success(f"{new_id} · {_t('Rule saved.')}")


def policy_history_tab(editor: PolicyEditor) -> None:
    st.caption(_t("Policy changes appear newest first with the editor and before/after values."))
    history = editor.history(limit=50)
    if not history:
        st.info(_t("No policy changes have been recorded."))
        return
    history_frame = pd.DataFrame([{
        "when": row.get("created_at", ""), "editor": row.get("editor", ""),
        "kind": row.get("kind", ""), "rule": row.get("rule_id", ""),
        "version": f"{row.get('version_before', '')} → {row.get('version_after', '')}",
        "changes": "; ".join(row.get("changes") or []),
    } for row in history])
    if not history_frame.empty:
        history_frame["event_time"] = pd.to_datetime(history_frame["when"],
                                                       errors="coerce", utc=True)
        timeline = history_frame.dropna(subset=["event_time"]).copy()
        if not timeline.empty:
            timeline["policy event"] = timeline["kind"].astype(str) + " · " + timeline["rule"]
            figure = px.scatter(timeline, x="event_time", y="policy event",
                                color="kind", hover_data=["editor", "version"],
                                color_discrete_sequence=["#ff7180", "#718fe6",
                                                         "#40d6ae", "#ffb454"])
            figure.update_layout(xaxis_title="", yaxis_title="")
            _chart(figure, "policy-change-timeline", 280)
    with st.expander(_t("Change records")):
        display_history = history_frame.drop(columns=["event_time"], errors="ignore").rename(
            columns={"when": _t("Time"), "editor": _t("Editor"), "kind": _t("Change type"),
                     "rule": _t("Rule"), "version": _t("Policy version"),
                     "changes": _t("Changes")})
        st.dataframe(display_history,
                     use_container_width=True, hide_index=True)

    chosen = st.selectbox(_t("Inspect one change"), [row["edit_id"] for row in history],
                          format_func=lambda edit_id: next(
                              (f"#{row['edit_id']} {row['rule_id']} "
                               f"{row.get('created_at', '')}" for row in history
                               if row["edit_id"] == edit_id), str(edit_id)))
    row = next(row for row in history if row["edit_id"] == chosen)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{_t('Before')}**")
        st.json(row.get("before") or {})
    with right:
        st.markdown(f"**{_t('After')}**")
        st.json(row.get("after") or {})


def policy_document_tab() -> None:
    """Summarise, or ask questions about, the document that was just released.

    It reads the released text only (the same string "Safe version as text" shows on the
    Review page), never the original. It cannot change a decision or the release.
    """
    analysis = st.session_state.get("analysis")
    last = st.session_state.get("last")
    if analysis is None or not last:
        st.info(_t("Create a protected copy first to use the document assistant."))
        return

    st.subheader(_t("Document assistant"))
    released = safe_text(analysis.document, last["applied"]["changed"])
    run_id = last["run_id"]
    if st.session_state.get("doc_qa_run") != run_id:       # a new release: a new conversation
        st.session_state["doc_qa_run"] = run_id
        st.session_state["doc_qa_history"] = []
    history: list[dict] = st.session_state["doc_qa_history"]

    st.caption(f"{_t('Protected copy')}: {analysis.document.filename} · "
               f"{len(released):,} {_t('characters')}. {_t('The assistant only sees the protected copy, never the original.')}")
    if config.resolve_provider() == "api":
        st.caption(_t("The document text may be processed by an external service."))
    else:
        st.caption(_t("The document text stays on this device."))
    if last["applied"]["incomplete"]:
        st.warning(f"{len(last['applied']['pending'])} {_t('items need your review')}. "
                   f"{_t('They were excluded from this protected copy.')}")

    ready = engine.status().get("installed", False)
    if not ready:
        st.info(_t("Document questions are temporarily unavailable."))

    question = st.text_input(_t("Ask about this released document"), key="doc_qa_question",
                             placeholder=_t("Who signed the agreement? / من الطرف الثاني في العقد؟"))
    ask_col, sum_col, clear_col = st.columns([1, 1, 3])
    ask = ask_col.button(_t("Ask"), type="primary", key="doc_qa_ask",
                         disabled=not (ready and question.strip()))
    summarize = sum_col.button(_t("Summarize released document"), key="doc_qa_summarize", disabled=not ready)
    if clear_col.button(_t("Clear messages"), key="doc_qa_clear", disabled=not history):
        history.clear()
        st.rerun()

    if ask or summarize:
        with st.spinner(_t("Reading the protected copy…")):
            try:
                history.append(document_qa.summarize(released) if summarize
                               else document_qa.ask(released, question))
            except (document_qa.DocumentQAError, engine.ModelNotInstalled) as exc:
                st.error(_t("The document question could not be answered."))
                with st.expander(_t("Technical details")):
                    st.code(str(exc))
            except Exception as exc:                                   # noqa: BLE001
                st.error(_t("The document question could not be answered."))
                with st.expander(_t("Technical details")):
                    st.code(str(exc))

    for entry in reversed(history):
        with st.container(border=True):
            st.markdown(f"**{_t('Summary')}**" if entry["kind"] == "summary"
                        else f"**{_t('Question')}:** {entry['question']}")
            show_text(entry["answer"])
            for note in entry["notes"]:
                st.caption(note)
            with st.expander(_t("Response details")):
                st.caption(f"{_t('Model name')}: {entry['model']} · "
                           f"{_t('Inference time')}: {entry['seconds']}s")


# --------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="Sitr | سِتر — Privacy workspace",
                       page_icon="🔒", layout="wide", initial_sidebar_state="auto")
    st.session_state.setdefault("ui_language", "en")
    _apply_theme()

    with st.sidebar:
        st.markdown(
            f'<div style="padding:.9rem .35rem 1.1rem">'
            f'<div class="brand-lockup" dir="ltr" aria-label="{html.escape(PRODUCT_BRAND)}">'
            '<span class="brand-latin">Sitr</span><span class="brand-divider">|</span>'
            '<bdi class="brand-arabic" dir="rtl">سِتر</bdi></div>'
            f'<div class="brand-tagline">{html.escape(_t("Privacy workspace"))}</div>'
            f'<div class="brand-tagline">{html.escape(_t("Private data. Purposeful sharing."))}</div></div>',
            unsafe_allow_html=True)
        st.caption(_t("Language"))
        st.segmented_control(
            _t("Language"), options=("en", "ar"), key="ui_language",
            format_func=lambda value: "EN" if value == "en" else "عربي",
            label_visibility="collapsed")
        pages = ["Upload", "Review", "Audit log", "Policy"]
        current_page = st.session_state.get("page", "Upload")
        if current_page not in pages:
            current_page = "Upload"
        page = st.radio(_t("Workspace"), pages, index=pages.index(current_page),
                        format_func=_t, label_visibility="collapsed")
        st.session_state["page"] = page
        st.divider()
        st.caption(_t("Audit values are masked. Chart data stays local."))

    {"Upload": page_upload, "Review": page_review, "Audit log": page_audit,
     "Policy": page_policy}[page]()


if __name__ == "__main__":
    main()
