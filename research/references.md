# المراجع الأولية

تمت مراجعة صفحات عامة وملفات README/صفحات المشاريع بتاريخ 2026-10-23. أوصاف القدرات والتراخيص هنا من المصادر المشار إليها؛ يجب فحص LICENSE الفعلي والاعتماديات قبل نقل الشيفرة.

## أبحاث وفهارس

1. **Self-Improvements in Modern Agentic Systems — Survey Hub**  
   https://selfimproving-agent.github.io/  
   خريطة بحثية لآليات تحسين النموذج والذاكرة والمهارات والأدوات وبنية الوكيل والتقييم.

2. **The Path to Recursive Self-Improving Agents**  
   https://self-improving-agent.com/  
   مراجعة بحثية عن تحويل الخبرة والتقييم إلى تغييرات مستمرة في مكونات الوكيل.

3. **Awesome Self-Evolving Agents**  
   https://github.com/sukoji/awesome-self-evolving-agents  
   فهرس لمشاريع وأوراق ذات صلة.

4. **Awesome Self-Improving Agent**  
   https://github.com/interactivetech/awesome-self-improving-agent  
   فهرس بحثي يركز على التغييرات القابلة لإعادة الاستخدام عبر المهام والجلسات.

## مستودعات مرجعية

5. **RSIAgent**  
   https://github.com/razor-ai/rsiagent  
   الاستكشاف واسع ثم عميق، Actor وVerifier مستقل، ذاكرة قابلة لإعادة الاستخدام، وتقييم منفصل.

6. **EvoSkill**  
   https://github.com/sentient-agi/EvoSkill  
   توليد بدائل للمهارات والتعليمات وتقييمها على بيانات تحقق، مع حفظ نسخ وفروق.

7. **SkillMem**  
   https://github.com/liza-studio/skillmem  
   ذاكرة مهارات محلية مع الاسترجاع والتعزيز والاضمحلال وتسجيل المصدر.

8. **POLYROB**  
   https://github.com/theselfruleorg/polyrob  
   وكيل ذاتي الاستضافة مع ذاكرة طويلة الأجل ومهارات ومسار فحص/موافقة.

9. **Self-Evolving Coding Agents Survey Repository**  
   https://github.com/iSEngLab/Awesome-Self-Evolving-Coding-Agents  
   خريطة لأنظمة تتطور فيها الذاكرة والمهارات والأدوات وبنية الوكيل والأوزان.

## مشروعنا الحالي

10. **KASR-1 — المستودع**  
    https://github.com/arkanws513-spec/Kasr-1  
    كان يتضمن نموذج Transformer صغيرًا من التهيئة العشوائية، وtokenizer على مستوى UTF-8 bytes، وتدريبًا واستدلالًا وذاكرة محادثة. لم يكن لديه checkpoint مدرّب عند checkout جديد بحسب README السابق. فرع الأرشيف يحفظ تلك النسخة قبل إعادة التأسيس:
    https://github.com/arkanws513-spec/Kasr-1/tree/archive/pre-rebuild-2026-10-23

## ملاحظات ترخيص وتشغيل

- Apache-2.0 في RSIAgent وEvoSkill بحسب مستودعاتهما؛ احتفظ بإشعارات الترخيص إذا نُقلت شيفرة فعلية.
- لم نعتمد على نسخ شيفرة SkillMem أو POLYROB في هذا التأسيس؛ يجب مراجعة ترخيص كل نسخة واعتمادياتها قبل أي استخدام.
- ترخيص الكود لا يمنح تلقائيًا حقوق استخدام بيانات التدريب أو أوزان النماذج أو benchmark assets التابعة لجهة أخرى.
- لا نعد أي مشروع «مجانيًا بالكامل» لمجرد أن مستودعه مفتوح المصدر؛ قد يتطلب inference مزودًا خارجيًا أو GPU أو بنية تشغيل مدفوعة.
