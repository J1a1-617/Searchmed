const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'SearchAgent Evaluation';
pptx.subject = 'Case evidence value for predictive clinical benchmark';
pptx.title = '病例证据是否帮助 Agent 回答临床预测问题';
pptx.company = 'THUNLP';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'PingFang SC', bodyFontFace: 'PingFang SC', lang: 'zh-CN'
};
pptx.defineSlideMaster({
  title: 'MASTER',
  background: { color: 'F5F7FA' },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 0.10, fill: { color: '13B8A6' }, line: { color: '13B8A6' } } },
    { text: { text: 'SEARCHAGENT · CASE EVIDENCE AUDIT', options: { x: 0.55, y: 7.12, w: 5.5, h: 0.16, fontFace: 'Aptos', fontSize: 8, color: '718096', margin: 0 } } },
    { text: { text: '2026.08', options: { x: 11.8, y: 7.12, w: 0.9, h: 0.16, fontFace: 'Aptos', fontSize: 8, color: '718096', align: 'right', margin: 0 } } },
  ],
  slideNumber: { x: 12.75, y: 7.10, color: '718096', fontSize: 8 },
});

const C = { navy:'15324A', teal:'13B8A6', cyan:'4DA3C7', red:'D95D5D', amber:'E4A83A', ink:'23303D', gray:'607080', light:'E9EEF3', white:'FFFFFF', green:'2E9D71', purple:'7057C7' };
const OUT = path.join(__dirname, '..', 'benchmark_results', 'agent_case_value_review_20260805.pptx');
const SCRIPT = path.join(__dirname, '..', 'benchmark_results', 'agent_case_value_review_20260805_speaker_script.md');
const EXACT = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'benchmark_results', 'ab3_exact_model_inputs.json'), 'utf8'));
const notes = [];

function addTitle(slide, title, subtitle='') {
  slide.addText(title, { x:0.58,y:0.34,w:12.1,h:0.46,fontFace:'PingFang SC',fontSize:25,bold:true,color:C.navy,margin:0,breakLine:false,fit:'shrink' });
  if (subtitle) slide.addText(subtitle, { x:0.60,y:0.86,w:11.9,h:0.28,fontSize:10.5,color:C.gray,margin:0,fit:'shrink' });
}
function box(slide,x,y,w,h,title,body,color=C.teal) {
  slide.addShape(pptx.ShapeType.roundRect,{x,y,w,h,rectRadius:0.06,fill:{color:C.white},line:{color:'D7E0E8',width:1},shadow:{type:'outer',color:'B8C4CF',opacity:0.15,blur:1,angle:45,distance:1}});
  slide.addShape(pptx.ShapeType.rect,{x,y,w:0.08,h,fill:{color},line:{color}});
  slide.addText(title,{x:x+0.22,y:y+0.18,w:w-0.38,h:0.28,fontSize:14,bold:true,color:C.navy,margin:0,fit:'shrink'});
  slide.addText(body,{x:x+0.22,y:y+0.58,w:w-0.38,h:h-0.72,fontSize:11,color:C.ink,breakLine:false,margin:0.02,fit:'shrink',valign:'top',bullet:undefined});
}
function tag(slide,text,x,y,w,color=C.teal) {
  slide.addShape(pptx.ShapeType.roundRect,{x,y,w,h:0.34,rectRadius:0.05,fill:{color,transparency:4},line:{color,transparency:100}});
  slide.addText(text,{x:x+0.08,y:y+0.075,w:w-0.16,h:0.17,fontSize:9,bold:true,color:C.white,align:'center',margin:0,fit:'shrink'});
}
function arrow(slide,x,y,w,color=C.teal) {
  slide.addShape(pptx.ShapeType.chevron,{x,y,w,h:0.44,fill:{color,transparency:10},line:{color,transparency:100}});
}
function addSpeaker(slide, title, text) {
  const topic = title.replace(/^第\d+页｜/, '');
  notes.push(`## 第${notes.length + 1}页｜${topic}\n\n${text}\n`);
  if (slide.addNotes) slide.addNotes(text.split('\n'));
}
function rowsTable(slide, rows, x,y,w,h, widths) {
  slide.addTable(rows,{x,y,w,h,border:{type:'solid',color:'D9E2EA',pt:0.7},fill:C.white,color:C.ink,fontSize:10,margin:0.08,rowH:0.45,colW:widths,autoFit:false,
    bold:false, valign:'mid', breakLine:false,
  });
}

// 1
{
  const s=pptx.addSlide('MASTER');
  s.background={color:C.navy};
  s.addShape(pptx.ShapeType.rect,{x:0,y:0,w:13.333,h:7.5,fill:{color:C.navy},line:{color:C.navy}});
  s.addShape(pptx.ShapeType.arc,{x:8.8,y:-1.2,w:5.2,h:5.2,adjustPoint:0.32,rotate:18,fill:{color:C.teal,transparency:15},line:{color:C.teal,transparency:100}});
  s.addShape(pptx.ShapeType.arc,{x:9.7,y:3.8,w:4.2,h:4.2,adjustPoint:0.25,rotate:210,fill:{color:C.cyan,transparency:30},line:{color:C.cyan,transparency:100}});
  s.addText('病例证据，到底有没有\n帮助 Agent 答题？',{x:0.72,y:1.12,w:8.9,h:1.55,fontSize:34,bold:true,color:C.white,margin:0,breakLine:false,fit:'shrink'});
  s.addText('基于真实 benchmark 病例、真实检索证据与 Golden Context 消融的逐题审计',{x:0.75,y:2.95,w:8.7,h:0.48,fontSize:16,color:'CDE7E4',margin:0,fit:'shrink'});
  s.addShape(pptx.ShapeType.line,{x:0.75,y:3.65,w:5.4,h:0,line:{color:C.teal,width:3}});
  s.addText('结论先行',{x:0.75,y:4.05,w:1.4,h:0.28,fontSize:13,bold:true,color:C.teal,margin:0});
  s.addText('有帮助，但仅在 case 与目标药物、治疗阶段、结局维度和时间窗足够直接时稳定成立。当前瓶颈主要在证据适配与证据作用方式，而不是 Generate 的理解能力。',{x:0.75,y:4.45,w:8.3,h:1.05,fontSize:20,bold:true,color:C.white,margin:0,fit:'shrink'});
  s.addText('样本：3 个可审计节点 · 对照：Question-only / Agent / Golden Context',{x:0.75,y:6.35,w:8.6,h:0.3,fontSize:11,color:'9FB7C8',margin:0});
  addSpeaker(s,'第1页｜标题',`今天只回答一个问题：检索到的病例证据，是否真的让 Agent 比直接把问题交给 GPT-5 更会答题。这里不讨论存在日期解析偏差的 M7，也不使用未运行 Judge 时的 Composite。我们只看逐病例的真实输出、真实进入 Generate 的证据，以及 Golden Context 上限实验。先说结论：病例证据确实可以帮助，但前提是病例与目标问题在药物、治疗阶段、结局维度和时间窗上足够直接。当前主要瓶颈不在 Generate，而在证据是否适配，以及 Context 如何使用弱证据。`);
}

// 2
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'1｜Benchmark 是怎样形成的','真实患者时间线被切成多个“治疗决策节点”，预测节点后 8–12 周结局');
 box(s,0.62,1.34,2.55,1.45,'44 例真实患者','NSCLC / SCLC / 结直肠癌 / 卵巢癌\n包含脑转移或脑膜转移',C.navy);
 arrow(s,3.37,1.82,0.48);
 box(s,4.00,1.34,2.55,1.45,'135 个决策节点','同一患者在不同换药、加量、联合或局部治疗节点形成多道题',C.cyan);
 arrow(s,6.75,1.82,0.48);
 box(s,7.38,1.34,2.55,1.45,'输入：切点前信息','疾病、突变、既往治疗、当前状态，以及医生实际采用的方案',C.teal);
 arrow(s,10.13,1.82,0.48);
 box(s,10.76,1.34,1.95,1.45,'GT','节点后 8–12 周真实随访',C.green);
 s.addText('每个节点的任务',{x:0.68,y:3.35,w:2.2,h:0.32,fontSize:17,bold:true,color:C.navy,margin:0});
 s.addText('“截至该治疗开始时点，只使用当时可见的信息，预测医生实际采用方案在未来 8–12 周的结局。”',{x:0.68,y:3.88,w:11.9,h:0.66,fontSize:21,bold:true,color:C.ink,margin:0,fit:'shrink'});
 const rows=[
  [{text:'输出维度',options:{bold:true,color:C.white,fill:C.navy}},{text:'真实随访标签',options:{bold:true,color:C.white,fill:C.navy}},{text:'本报告是否采用',options:{bold:true,color:C.white,fill:C.navy}}],
  ['总体净获益','明显 / 有限稳定 / 无明显 / 进展有害','采用'],
  ['影像学','体部 RECIST、CNS/脑膜 RECIST','采用'],
  ['临床轨迹','CSF、症状、最高毒性等级','采用'],
  ['M7 / LLM Judge / Composite','时间解析或实验条件存在偏差','不用于结论'],
 ]; rowsTable(s,rows,0.68,4.82,11.95,1.75,[2.3,5.2,4.45]);
 addSpeaker(s,'第2页｜Benchmark形成方式',`这个 benchmark 来自四十四例真实肿瘤患者，共拆成一百三十五个治疗决策节点。同一个患者可以在换药、剂量上调、联合治疗或者局部治疗时形成不同问题。每道题只给时间切点之前可见的疾病背景、分子特征、既往治疗、当前状态，以及医生实际采用的治疗方案。标准答案来自节点之后八到十二周的真实随访。这里我们只使用总体获益、体部和中枢 RECIST、症状、脑脊液与毒性这些直接标签，不使用已知存在日期格式偏差的 M7，也不使用当前没有统一 Judge 的 Composite。`);
}

// Data & KB 1
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'2A｜目前收集了什么数据','780篇肿瘤文献经过三级加工；结构化层覆盖417篇，其余仍以文献证据形式保留');
 const xs=[0.66,4.57,8.48], colors=[C.gray,C.cyan,C.teal];
 const titles=['Step 1 · 全文抽取层','Step 2 · 临床清洗层','Step 3 · 结构化病例层'];
 const bodies=[
  '780篇\n从PMC/HTML经Trafilatura抽取\n标题、作者、日期、PMID、正文、source_file\n约30 MB',
  '780篇\nLLM判断是否为case report\n保留Abstract + Case Presentation等临床正文\n约4.7 MB',
  '417篇\n基线画像、时间线事件、用药、疗效、进展、毒性、DDI与结局摘要\n约2.0 MB'
 ];
 for(let i=0;i<3;i++){box(s,xs[i],1.35,3.30,2.50,titles[i],bodies[i],colors[i]); if(i<2) arrow(s,xs[i]+3.41,2.30,0.34,'AAB7C2');}
 s.addText('额外结构化知识表',{x:0.72,y:4.30,w:2.6,h:0.32,fontSize:17,bold:true,color:C.navy,margin:0});
 const rows=[
  [{text:'知识表',options:{bold:true,color:C.white,fill:C.navy}},{text:'规模',options:{bold:true,color:C.white,fill:C.navy}},{text:'内容',options:{bold:true,color:C.white,fill:C.navy}},{text:'作用',options:{bold:true,color:C.white,fill:C.navy}}],
  ['药名映射','23条','通用名、商品名、别名','归一化和扩展药名'],
  ['DDI规则','117条','酶/转运体、角色、相互作用可能性','组合用药风险检索'],
  ['药物PK关系','299条','ADME、底物/抑制/诱导、相关药物','机制与PK扩展'],
  ['突变—药物','609条','突变及关联药物','从分子特征扩展候选药物'],
 ]; rowsTable(s,rows,0.70,4.72,11.95,1.84,[2.0,1.0,5.05,3.9]);
 addSpeaker(s,'数据类型',`目前知识库的主体来自七百八十篇肿瘤相关文献，并经过三级加工。Step 1 是全文抽取层，保存标题、作者、日期、PMID、来源文件和抽取正文，主要用于保留最大的信息覆盖。Step 2 是临床清洗层，由模型判断是否为病例报告，并保留摘要、病例经过和治疗描述等低噪声正文。Step 3 是结构化病例层，目前覆盖四百一十七篇，抽取基线画像、时间线事件、药物、疗效、进展、毒性和结局。此外还有药名映射、DDI规则、药代关系和突变药物表，用于实体归一化与检索扩展。需要注意，Step 3覆盖率不是百分之百，因此未结构化的文献仍通过Step 1和Step 2文本参与检索。`);
}

// Data & KB 2
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'2B｜KB 不是一个表，而是“病例证据 + 多索引 + 规则图谱”','同一份文献同时保留可筛选字段、语义片段、关键词片段和可追溯原文');
 box(s,0.62,1.30,2.52,1.48,'Structured KB','SQLite结构化库\n癌种、组织学、分期、突变、药物、事件、疗效、毒性、时间线',C.navy);
 box(s,0.62,3.10,2.52,1.48,'Dense KB','case_semantic：20,441 chunks\nevent_semantic：1,615 chunks\nstructured_semantic：417 chunks',C.cyan);
 box(s,0.62,4.90,2.52,1.35,'Sparse KB','BM25：23,521 docs\n适合药名、突变、剂量、缩写和罕见关键词',C.teal);
 s.addShape(pptx.ShapeType.chevron,{x:3.48,y:2.65,w:0.62,h:1.25,fill:{color:'B7C3CD'},line:{color:'B7C3CD'}});
 box(s,4.38,1.30,3.50,2.00,'Hybrid + RRF','Structured精确约束\nDense语义近邻\nBM25词法命中\n多路归一化与融合后产生候选',C.purple);
 box(s,4.38,3.72,3.50,1.72,'LLM Rerank + Review','按本轮最细目标重排\n识别支持、反证和风险\n区分direct / partial / analog / counter',C.amber);
 s.addShape(pptx.ShapeType.chevron,{x:8.18,y:2.65,w:0.62,h:1.25,fill:{color:'B7C3CD'},line:{color:'B7C3CD'}});
 box(s,9.08,1.30,3.55,2.00,'Fetch Evidence','候选定位符必须回源\n返回真实chunk正文、PMID、日期、source_file和field_path',C.green);
 box(s,9.08,3.72,3.55,1.72,'最终证据记忆','claim绑定chunk_id\nContext选择进入Generate的证据\nCitation可追溯到原文件',C.teal);
 s.addShape(pptx.ShapeType.roundRect,{x:3.98,y:5.82,w:5.05,h:0.72,fill:{color:'E7F6F3'},line:{color:'B7E2DA'}});
 s.addText('KB形态：不是生成式“百科”，而是带时间线和出处的可检索病例证据仓库。',{x:4.25,y:6.04,w:4.52,h:0.26,fontSize:12.5,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
 addSpeaker(s,'KB形态',`当前 KB 不是一个单一数据库，而是病例证据、多种索引和规则知识表的组合。结构化 SQLite 用于癌种、组织学、突变、药物、疗效、毒性和时间线的精确过滤。Dense KB 包含两万零四百四十一条病例正文片段、一千六百一十五条事件片段和四百一十七条结构化语义表示，用于语义近邻检索。BM25包含两万三千五百二十一条文档，擅长精确药名、突变、剂量和缩写。多路候选融合后进入 LLM Rerank 和 Evidence Review，最后必须通过 Fetch Evidence 回到真实正文，并用 chunk ID、PMID、日期和 source file 保持可追溯。`);
}

// Data & KB 3
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'2C｜这些信息对答题有什么帮助','不同数据层解决不同问题；“找到相关内容”不等于“足以预测当前结局”');
 const rows=[
  [{text:'信息类型',options:{bold:true,color:C.white,fill:C.navy}},{text:'可帮助的推断',options:{bold:true,color:C.white,fill:C.navy}},{text:'真实实验观察',options:{bold:true,color:C.white,fill:C.navy}},{text:'当前边界',options:{bold:true,color:C.white,fill:C.navy}}],
  ['病例全文 / Step2','治疗序列、症状、影像、CSF、剂量与个体结局','case_1找到同药同剂量LM反证，纠正症状方向','长文本噪声；治疗阶段可能混杂'],
  ['结构化事件 / Step3','快速匹配突变—药物—疗效—时间—毒性','能先定位文档，再回Step2找细节','417/780覆盖；抽取可能漏细节'],
  ['药名 / PK / DDI表','同义名扩展、同类药物、代谢与相互作用风险','多轮无结果时扩大搜索空间','不能直接预测患者8–12周疗效'],
  ['Dense + BM25','兼顾语义相似与精确实体命中','能找到医学相关case','相似不保证治疗线/结局/时间窗直接'],
  ['原文与Citation','约束幻觉、支持逐条核查','所有最终证据可绑定chunk','历史绝对路径在GPU可能不可直接打开'],
 ]; rowsTable(s,rows,0.62,1.30,12.06,3.68,[2.25,3.1,3.45,3.26]);
 box(s,0.70,5.35,3.52,1.15,'对疗效最有帮助','相同药物+剂量+治疗线+转移部位+早期结局的病例',C.green);
 box(s,4.88,5.35,3.52,1.15,'对安全最有帮助','基线器官功能、既往耐受、剂量调整、G3+毒性与停药病例',C.red);
 box(s,9.06,5.35,3.52,1.15,'当前最大缺口','病例多为医学相关，但与具体预测维度和8–12周时间窗不够直接',C.amber);
 addSpeaker(s,'信息对答题的帮助',`不同数据层对答题的价值不同。Step 2病例正文最适合恢复治疗顺序、症状、影像、脑脊液和剂量，case 一就是靠同药、同剂量、同脑膜转移的原文反证纠正了症状方向。Step 3事件适合快速锁定突变、药物、疗效和时间，但目前只覆盖四百一十七篇，并可能漏掉正文细节。药名、PK和DDI表适合实体扩展与安全机制，不应直接决定患者八到十二周疗效。Dense和BM25能找到医学相关病例，但相似性不保证治疗线、结局维度和时间窗直接匹配。当前最缺的是两类数据：一类是相同治疗情境下的早期结局病例，另一类是包含器官功能、既往耐受、三级以上毒性和停药信息的安全性病例。`);
}

// 3
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'2｜如何隔离“Generate能力”和“病例证据价值”','同一个 BenchmarkPredictionGenerator，只改变它看到的信息');
 const xs=[0.72,4.64,8.56], colors=[C.gray,C.teal,C.purple];
 const titles=['A · Question-only','B · Agent','C · Golden Context'];
 const bodies=['只给原始病例和实际方案\n不提供检索证据\n测模型自身医学先验','完整检索链路\nEvidence Review → Memory → Context\n测 case evidence 的净增益','把完整真实随访标签明确给同一个 Generate\n测 Generate 是否会损失正确信息'];
 for(let i=0;i<3;i++){box(s,xs[i],1.45,3.35,2.05,titles[i],bodies[i],colors[i]); if(i<2) arrow(s,xs[i]+3.50,2.20,0.46,'AAB7C2');}
 s.addShape(pptx.ShapeType.roundRect,{x:1.05,y:4.18,w:11.1,h:1.55,rectRadius:0.06,fill:{color:'E7F6F3'},line:{color:'B7E2DA',width:1}});
 s.addText('Golden Context 结果',{x:1.35,y:4.52,w:2.5,h:0.32,fontSize:16,bold:true,color:C.teal,margin:0});
 s.addText('总体获益、体部 RECIST、CNS RECIST、症状方向、严重毒性：三题全部正确',{x:3.35,y:4.48,w:7.9,h:0.42,fontSize:18,bold:true,color:C.navy,margin:0,fit:'shrink'});
 s.addText('解释：Generate 能忠实消费正确上下文；Agent 的差异主要来自上游检索、证据角色与 Context，而不是输出 schema 本身。',{x:1.35,y:5.08,w:10.3,h:0.3,fontSize:12,color:C.ink,margin:0,fit:'shrink'});
 s.addText('注意：Golden Context 故意暴露随访答案，仅用于诊断 Generate 上限，不是实际系统成绩。',{x:1.05,y:6.20,w:11.1,h:0.3,fontSize:10.5,italic:true,color:C.red,align:'center',margin:0});
 addSpeaker(s,'第3页｜实验设计与Golden Context',`为了判断到底是 Generate 的 prompt 有问题，还是上游证据有问题，我们让完全相同的 BenchmarkPredictionGenerator 接受三种输入。第一种只给问题，代表模型自身医学先验。第二种给 Agent 检索和 Context。第三种把完整真实随访标签直接给 Generate，作为 Golden Context 上限实验。结果是 Golden Context 下，三道题的总体获益、体部 RECIST、中枢 RECIST、症状和严重毒性全部正确。这说明 Generate 能够忠实保留正确信息。它不是当前主要瓶颈。需要强调，Golden Context 是故意泄漏答案的诊断实验，不是实际系统成绩。`);
}

// 4
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'逐字输入 1/4｜三题共同的输出指令与 JSON Schema','这是 construct_inference_prompt() 的共同后半段；与后面任一病例前半段直接拼接，即为完整 benchmark prompt');
 const raw=EXACT['case_1_node_2'].benchmark_common_output_suffix;
 s.addShape(pptx.ShapeType.roundRect,{x:0.55,y:1.18,w:12.22,h:5.72,fill:{color:'101B25'},line:{color:'2C4355'}});
 s.addText(raw,{x:0.78,y:1.38,w:11.76,h:5.30,fontFace:'Menlo',fontSize:7.5,color:'E8F0F5',margin:0.02,breakLine:false,fit:'shrink',valign:'top'});
 s.addText('逐字展示 · 无摘要 · 无省略',{x:10.20,y:0.83,w:2.48,h:0.22,fontSize:9,bold:true,color:C.teal,align:'right',margin:0});
 addSpeaker(s,'第4页｜完整prompt的共同后半段',`这一页是三道题共同的逐字输出指令和 JSON Schema，没有摘要，也没有省略。后面每一道题的病例专属前半段，与这一页内容直接拼接，就是 construct_inference_prompt 返回的完整 benchmark prompt。模型被要求输出总体获益、体部和中枢 RECIST、脑脊液、症状、毒性、置信度、理由和证据。`);
}
for (const [idx, cid] of ['case_1_node_2','case_3_node_2','case_5_node_1'].entries()) {
 const s=pptx.addSlide('MASTER'); addTitle(s,`逐字输入 ${idx+2}/4｜${cid} 的病例专属 prompt`,`与上一页共同输出指令/schema拼接后，即为该题完整 benchmark input；以下逐字、无省略`);
 const raw=EXACT[cid].benchmark_case_specific_prefix;
 s.addShape(pptx.ShapeType.roundRect,{x:0.55,y:1.18,w:12.22,h:5.72,fill:{color:'101B25'},line:{color:'2C4355'}});
 s.addText(raw,{x:0.78,y:1.38,w:11.76,h:5.30,fontFace:'Menlo',fontSize:8.5,color:'E8F0F5',margin:0.02,breakLine:false,fit:'shrink',valign:'top'});
 s.addText('逐字展示 · ground_truth 不在输入中',{x:9.55,y:0.83,w:3.12,h:0.22,fontSize:9,bold:true,color:C.teal,align:'right',margin:0});
 const page=idx+5;
 addSpeaker(s,`第${page}页｜${cid}逐字benchmark输入`,`这一页逐字展示 ${cid} 的病例专属 benchmark prompt 前半段，包括任务说明、时间切点、诊断、转移、突变、既往治疗、当前临床状态和实际采用方案。它没有经过摘要，也不包含 ground truth。把这一页与上一页共同的输出指令和 JSON Schema 拼接，就是模型收到的完整 benchmark prompt。完整字符串以及 Agent 最终 Generate 的 system prompt、user payload 和 function 名称，也另存于逐字输入附件。`);
}

// 8 overall
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'3｜总体观察：病例证据有用，但不稳定','三题全部没有 direct key finding；实际进入 Generate 的是 counter 或 partial case');
 const rows=[
  [{text:'病例',options:{bold:true,color:C.white,fill:C.navy}},{text:'Question-only',options:{bold:true,color:C.white,fill:C.navy}},{text:'Agent变化',options:{bold:true,color:C.white,fill:C.navy}},{text:'证据价值结论',options:{bold:true,color:C.white,fill:C.navy}}],
  ['case_1_node_2','症状：部分改善','症状→无变化；总体/体部/CNS保持正确','明确帮助'],
  ['case_3_node_2','有限稳定 / SD / 无变化','核心判断不变，仍漏掉G3毒性与停药','基本无帮助'],
  ['case_5_node_1','明显获益；CNS PR','CNS→SD正确；总体→有限稳定错误','局部帮助、总体偏负'],
 ]; rowsTable(s,rows,0.67,1.42,12.0,2.5,[2.05,3.15,3.85,2.95]);
 s.addText('关键审计事实',{x:0.75,y:4.35,w:2.0,h:0.3,fontSize:17,bold:true,color:C.navy,margin:0});
 box(s,0.72,4.82,3.65,1.26,'0 / 3','三题的 AnswerContext.key_findings 均为空：没有一题获得真正 direct supporting evidence。',C.red);
 box(s,4.82,4.82,3.65,1.26,'1 / 3','一题由高度相关 counter case 成功纠正症状方向。',C.green);
 box(s,8.92,4.82,3.65,1.26,'2 / 3','两题证据与目标结局不够直接：一个无增益，一个发生过度保守。',C.amber);
 s.addText('所以当前实验回答的是：弱病例证据能否帮助，而不是“直接命中病例答案”能否帮助。',{x:0.72,y:6.45,w:11.9,h:0.34,fontSize:14,bold:true,color:C.ink,align:'center',margin:0});
 addSpeaker(s,'第8页｜总体观察',`先看总体结果。三道题中，第一题 Agent 明确帮助，第二题基本没有帮助，第三题局部帮助但总体偏负。更重要的是，三题的 key findings 全部为空，也就是说没有一题得到真正的直接支持证据。最后进入 Generate 的都是反证或者部分匹配病例。所以这个实验实际测试的是弱病例证据能不能帮助，而不是直接命中标准病例是否有帮助。答案是：可以，但不稳定。`);
}

// 5 case1
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'4｜模型输入字段拆解：case_1_node_2','便于阅读的字段视图；完整逐字 prompt 已在前面的黑底附页展示');
 box(s,0.62,1.22,3.78,1.42,'时间切点与预测任务','时间切点：2023-10-12\n预测医生实际采用方案在未来8–12周的总体获益、体部/CNS RECIST、CSF、症状与毒性。',C.navy);
 box(s,0.62,2.86,3.78,1.58,'疾病背景','左肺下叶腺癌IV期\n转移：骨、脑膜\n主要驱动突变：EGFR 19del\n耐药/旁路/共突变：未提供',C.cyan);
 box(s,0.62,4.66,3.78,1.52,'实际采用方案','奥希替尼 160 mg，口服\n联合策略：单药治疗',C.teal);
 box(s,4.72,1.22,7.92,1.72,'既往治疗史','2022-04 至 2023-10：奥希替尼80 mg qd\n最佳疗效：PR\n调整原因：脑膜转移进展',C.amber);
 box(s,4.72,3.18,7.92,2.38,'当前临床状态','症状：头晕、肢体麻木、咀嚼费力、眼疲劳\n影像：新增右侧中央沟软脑膜改变，转移可能\n脑脊液：细胞学阴性，颅压正常\n体能状态：ECOG 2',C.purple);
 s.addShape(pptx.ShapeType.roundRect,{x:4.72,y:5.82,w:7.92,h:0.58,fill:{color:'FFF4DD'},line:{color:'EBCB87'}});
 s.addText('不可见：ground_truth、key_evidence_items、未来随访。直接GPT另看到输出schema；Agent检索query删除schema但保留以上临床事实。',{x:4.96,y:5.98,w:7.45,h:0.24,fontSize:11.2,bold:true,color:'7B5611',margin:0,fit:'shrink'});
 addSpeaker(s,'第9页｜case_1输入字段拆解',`前面的黑底页已经逐字展示完整 benchmark prompt。这一页只是为了讲解方便，把同一输入重新拆成字段。时间切点是二零二三年十月十二日。诊断是左肺下叶腺癌四期，骨和脑膜转移，EGFR 十九号外显子缺失。既往使用奥希替尼八十毫克，最佳疗效部分缓解，后来因脑膜转移进展调整。当前症状、影像、脑脊液和 ECOG 均按 benchmark 原字段展示。实际方案是奥希替尼一百六十毫克口服单药。`);
}

// 5 case1 result
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'4B｜真实案例 A：高相关反证成功纠错','case_1_node_2 · EGFR 19del 脑膜转移 · 奥希替尼 80→160 mg');
 box(s,0.62,1.25,3.45,1.62,'问题摘要（非完整prompt）','既往奥希替尼80 mg获得PR，后以脑膜转移进展；当前加量至160 mg单药。完整模型输入见前页。',C.navy);
 box(s,0.62,3.08,3.45,1.48,'真实随访','总体：有限获益或稳定\n体部：SD｜CNS：SD\n症状：无变化｜毒性：G1头晕',C.green);
 box(s,0.62,4.78,3.45,1.18,'答案变化','Question-only：症状“部分改善”\nAgent：症状“无变化” ✓',C.teal);
 box(s,4.42,1.25,8.22,2.18,'进入 Generate 的真实证据 ①','“The dose of osimertinib was increased to 160 mg once daily. After 4 months, owing to progressive dizziness and concern about worsening LMD… repeat CSF showed increased protein and higher EGFR/TP53 VAF.”\n\n同药物｜同剂量｜同脑膜转移｜直接报告神经症状与CSF变化',C.green);
 box(s,4.42,3.73,8.22,1.92,'进入 Generate 的真实证据 ②','“After osimertinib + capmatinib, chest lesions shrank, but neurologic symptoms progressed owing to worsening LMD; adding bevacizumab provided no meaningful neurologic improvement.”\n\n结局维度直接，但方案为联合治疗，适配性低于证据①',C.amber);
 s.addShape(pptx.ShapeType.roundRect,{x:4.42,y:5.95,w:8.22,h:0.65,fill:{color:'E7F6F3'},line:{color:'B7E2DA'}});
 s.addText('为什么有效：证据没有“给出答案”，但精确反驳了“加量后症状自然改善”的过度乐观先验。',{x:4.66,y:6.15,w:7.75,h:0.24,fontSize:13,bold:true,color:C.navy,margin:0,fit:'shrink'});
 addSpeaker(s,'第10页｜案例A结果与证据',`第一题是最清楚的正面案例。患者是 EGFR 十九号外显子缺失、脑膜转移，在奥希替尼八十毫克后进展，现在加量到一百六十毫克。只给问题时，模型预测症状部分改善。Agent 检索到一个同药、同剂量、同脑膜转移的病例：加量后四个月头晕继续加重，脑脊液蛋白和突变频率升高。还检索到一个联合治疗病例，胸部病灶缩小但脑膜和神经症状继续恶化。最终 Agent 把症状从部分改善修正成无变化，与真实随访一致。这说明高度相关的反证 case 能够有效校正模型的过度乐观先验。`);
}

// 6 case3
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'5｜模型输入字段拆解：case_3_node_2','完整逐字 prompt 已展示；本页突出治疗前缺失的毒性风险信息');
 box(s,0.62,1.22,3.78,1.42,'时间切点与预测任务','时间切点：2023-08\n预测培美曲塞+铂类双药在未来8–12周的总体获益、RECIST、症状与毒性。',C.navy);
 box(s,0.62,2.86,3.78,1.58,'疾病背景','右肺腺癌，骨转移\n主要驱动突变：EGFR 19del\n共突变：TP53 Y220H\nCNS/CSF：未评估',C.cyan);
 box(s,0.62,4.66,3.78,1.52,'实际采用方案','培美曲塞 标准剂量，静脉\n铂类 标准剂量，静脉\n联合策略：双药化疗',C.teal);
 box(s,4.72,1.22,7.92,2.05,'既往治疗史','① 2022-08至2023-07：埃克替尼标准剂量，最佳SD，因胸水进展停药\n② 2023-07至2023-08：阿美替尼标准剂量，最佳PD，因反应欠佳停药',C.amber);
 box(s,4.72,3.52,7.92,1.72,'当前临床状态','症状字段：反应欠佳\n胸部CT：提示PD\n脑脊液：NA\nECOG：NA',C.purple);
 s.addShape(pptx.ShapeType.roundRect,{x:4.72,y:5.52,w:7.92,h:0.82,fill:{color:'FCE8E8'},line:{color:'E6A7A7'}});
 s.addText('关键缺失：治疗前没有肝肾功能、血常规、骨髓储备、既往化疗耐受性或明确ECOG。模型无法看到未来“严重不良反应→停药”。',{x:4.96,y:5.72,w:7.42,h:0.40,fontSize:11.5,bold:true,color:'8A3333',margin:0,fit:'shrink'});
 addSpeaker(s,'第11页｜case_3输入字段拆解',`完整逐字 prompt 已经在黑底页展示。这一页突出输入结构。时间切点是二零二三年八月。患者是右肺腺癌、骨转移、EGFR 十九号外显子缺失并有 TP53 Y220H 共突变。埃克替尼后胸水进展，阿美替尼仅一个月即 PD。当前胸部 CT 提示 PD，但 ECOG、肝肾功能、血常规、骨髓储备和既往化疗耐受信息没有提供。实际方案是培美曲塞加铂类双药。因此，未来严重不良反应和停药包含治疗前输入难以预测的个体事件。`);
}

// 6 case3 result
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'5B｜真实案例 B：case 与真正结局不直接，无法救回','case_3_node_2 · EGFR 19del + TP53 · TKI失败后培美曲塞+铂');
 box(s,0.62,1.24,3.42,1.54,'问题摘要（非完整prompt）','两代EGFR-TKI先后失败，计划培美曲塞+铂类双药。完整模型输入见前页。',C.navy);
 box(s,0.62,3.02,3.42,1.45,'真实随访','进展或有害｜体部PD\n症状加重｜G3不良反应\n因不耐受停止化疗',C.red);
 box(s,0.62,4.69,3.42,1.20,'Question-only ≈ Agent','两者均预测：有限稳定 / SD / 症状无变化 / G0\nAgent没有救回关键结局',C.gray);
 box(s,4.38,1.24,8.28,1.55,'检索证据 ①：较近但仍非目标患者','“Afatinib progression → two cycles cisplatin + pemetrexed → CT showed disease stability.”\n能说明双药可能短期SD，但没有TP53 Y220H、严重毒性或停药信息。',C.amber);
 box(s,4.38,3.02,8.28,1.55,'检索证据 ②：分子背景与方案混杂','获得性BRAF变异队列：多种含铂/培美曲塞±贝伐方案；化疗组ORR 23.5%、PFS 5个月。\n不是目标分子背景，也不是统一双药方案。',C.amber);
 box(s,4.38,4.80,8.28,1.28,'检索证据 ③：不能直接外推','顺铂+培美曲塞+吉非替尼三联获得PR。\n目标病例是双药化疗，且真实失败由严重不良反应和停药主导。',C.red);
 s.addText('判断：不是“没有医学内容”，而是 case 回答“化疗一般能否控病”，benchmark 实际考“该患者能否耐受并持续治疗”。',{x:0.78,y:6.40,w:11.75,h:0.36,fontSize:13.5,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
 addSpeaker(s,'第12页｜案例B结果与证据',`第二题说明了为什么“检索到相关论文”不等于“对当前推断有帮助”。目标患者在两代 TKI 失败后接受培美曲塞加铂，真实结果是三级不良反应、停药、快速进展。Agent 找到的 case 主要说明这类化疗在其他患者中可能达到稳定或部分缓解，还混入 BRAF 变异队列和联合吉非替尼的三联方案。它们回答的是“化疗一般有没有疾病控制”，而 benchmark 真正由“这个患者是否能耐受并持续治疗”决定。数据库缺少基线器官功能、骨髓储备、既往耐受性、实际剂量和停药风险，因此 Agent 没有救回这道题。`);
}

// 7 case5
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'6｜模型输入字段拆解：case_5_node_1','完整逐字 prompt 已展示；本页突出形成奥希替尼CNS疗效先验的字段');
 box(s,0.62,1.22,3.78,1.42,'时间切点与预测任务','时间切点：2020-07\n预测奥希替尼80 mg单药在未来8–12周的全身、CNS、症状与毒性。',C.navy);
 box(s,0.62,2.86,3.78,1.76,'疾病背景','肺腺癌IV期\n转移：脑、双肺、纵膈淋巴结、胸膜\nEGFR Exon19缺失\nT790M：未提供',C.cyan);
 box(s,0.62,4.84,3.78,1.34,'实际采用方案','奥希替尼80 mg qd\n口服单药靶向治疗',C.teal);
 box(s,4.72,1.22,7.92,1.72,'既往治疗史','2020-04至2020-07：吉非替尼250 mg qd\n既往效果：喘憋症状改善\n调整原因：脑转移进展',C.amber);
 box(s,4.72,3.18,7.92,2.18,'当前临床状态','症状：头晕、头痛、恶心呕吐\n影像：脑内多发异常信号，最大病灶约0.7×0.5 cm；小脑脑膜受累\n脑脊液：未评估\n体能状态：ECOG 2',C.purple);
 s.addShape(pptx.ShapeType.roundRect,{x:4.72,y:5.64,w:7.92,h:0.70,fill:{color:'E7F6F3'},line:{color:'B7E2DA'}});
 s.addText('模型可据EGFR 19del、吉非替尼后CNS进展和奥希替尼CNS活性形成先验，但看不到三个月后的SD与症状明显改善。',{x:4.96,y:5.84,w:7.40,h:0.34,fontSize:11.5,bold:true,color:C.navy,margin:0,fit:'shrink'});
 addSpeaker(s,'第13页｜case_5输入字段拆解',`完整逐字 prompt 已经在黑底页展示。这一页突出形成医学先验的字段。时间切点是二零二零年七月。患者是 EGFR 十九号外显子缺失肺腺癌四期，有脑、双肺、纵隔淋巴结和胸膜转移。吉非替尼治疗三个月后喘憋改善，但脑转移进展。当前有头晕、头痛、恶心呕吐，影像显示脑内多发小病灶和小脑脑膜受累，ECOG 二分。实际采用奥希替尼八十毫克单药。模型不会看到三个月后的影像稳定和症状改善。`);
}

// 7 case5 result
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'6B｜真实案例 C：局部修正正确，但总体判断被压低','case_5_node_1 · EGFR 19del · 吉非替尼后CNS进展 → 奥希替尼80 mg');
 box(s,0.62,1.24,3.45,1.50,'问题摘要（非完整prompt）','一代TKI后以脑转移为主进展，T790M未知；改用奥希替尼80 mg。完整模型输入见前页。',C.navy);
 box(s,0.62,2.98,3.45,1.40,'真实随访','总体：明显获益\n体部SD｜CNS SD\n症状明显改善｜G1乏力',C.green);
 box(s,0.62,4.62,3.45,1.48,'答案变化','Question-only：明显获益 / CNS PR\nAgent：有限稳定 ✕ / CNS SD ✓\n一得一失，总体偏负',C.amber);
 box(s,4.42,1.24,8.20,1.48,'有用但不够直接的证据','“EGFR 19del患者一线奥希替尼80 mg，症状迅速改善；12周胸部CT为PR。”\n支持早期有效，但不是吉非替尼后CNS进展，也没有直接颅内结局。',C.teal);
 box(s,4.42,2.98,8.20,1.48,'被错误当作早期反证的证据','多个病例描述奥希替尼先获益约11个月，随后出现MET扩增、C797S等获得性耐药。\n“长期后耐药”不等于“8–12周不会获益”。',C.red);
 box(s,4.42,4.72,8.20,1.18,'Context 的错误转换','证据不完全 → 不应自动降低疗效类别\n更合理：保持“可能明显获益”，把不确定性反映在 confidence 和 CNS PR/SD 上。',C.purple);
 s.addText('判断：数据库已有相关奥希替尼病例，但缺少“相同治疗线 + CNS结局 + 早期时间窗”；Evidence Review 的时间语义进一步放大了偏差。',{x:0.76,y:6.38,w:11.75,h:0.38,fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
 addSpeaker(s,'第14页｜案例C结果与证据',`第三题是混合结果。只给问题时，模型判断明显获益并预测中枢 PR。Agent 把中枢修正成真实的 SD，这是帮助；但同时把正确的总体明显获益降成有限稳定，这是伤害。检索到的一条病例支持奥希替尼早期有效，但它是一线治疗且主要报告胸部病灶。另一些病例描述治疗十一个月后发生 MET 或 C797S 耐药。系统把这些长期耐药材料当成早期疗效反证，这是时间语义错误。证据不够直接时，更合理的是降低置信度或者把 CNS 从 PR 调整为 SD，而不是自动降低总体获益类别。`);
}

// 8 causes
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'7｜有帮助时为什么有效；没帮助时究竟缺什么','“规模不足”与“case—问题关系不直接”需要分开诊断');
 box(s,0.68,1.28,3.72,2.10,'有帮助的必要条件','✓ 同目标药物与剂量\n✓ 同治疗线与既往耐药背景\n✓ 同转移部位（尤其CNS/LM）\n✓ 直接报告目标结局\n✓ 观察时间接近8–12周',C.green);
 box(s,4.80,1.28,3.72,2.10,'规模/覆盖不足','case_5：缺一代TKI后、T790M未知、CNS进展、奥希替尼80 mg的早期颅内/症状证据。\n扩大“正确类型”的case库会有帮助。',C.cyan);
 box(s,8.92,1.28,3.72,2.10,'关系本身不直接','case_3：普通化疗疗效病例不能预测个体严重毒性与停药。\n需要不同字段与不同风险库，单纯增加疗效case不够。',C.red);
 s.addText('当前证据链的主要失真点',{x:0.72,y:3.88,w:3.0,h:0.34,fontSize:17,bold:true,color:C.navy,margin:0});
 const stages=[['Query / Plan','未稳定拆成疗效、CNS、症状、毒性、停药五类目标'],['Retrieval','命中同药/同突变，但未保证治疗线、方案和时间窗'],['Rerank / Review','长期耐药可能被误判为早期反证；联合方案被高估'],['Context','弱证据的“不确定”被转换成“疗效更差”'],['Generate','能消费正确上下文；本身不是主要瓶颈']];
 stages.forEach((d,i)=>{const x=0.70+i*2.48; box(s,x,4.43,2.18,1.62,d[0],d[1],i===4?C.green:[C.amber,C.amber,C.red,C.red][i]); if(i<4) arrow(s,x+2.22,5.00,0.22,'B5C1CB');});
 addSpeaker(s,'第15页｜原因拆解',`病例证据有用需要五个条件：同药物和剂量、同治疗线和耐药背景、同转移部位、直接报告目标结局、观察时间接近八到十二周。没有帮助时要区分两种原因。case 五主要是正确类型的病例覆盖不足，扩大 CNS 早期结局库会有帮助。case 三则是病例和目标推断关系本身不直接，普通化疗疗效病例不能预测个体严重毒性和停药，需要新的毒性字段和风险库。沿工作流看，Generate 能消费正确上下文，主要失真发生在目标拆分、检索适配、时间语义和 Context 融合。`);
}

// 9 roadmap
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'8｜优先改什么：让 case 只修改它真正支持的维度','从“总体保守化”改为“按结局维度的证据更新”');
 const y0=1.30;
 const items=[
  ['P0','证据作用域','每条证据绑定 outcome_dimension：overall / body / CNS / symptom / toxicity / discontinuation。',C.red],
  ['P0','时间语义','显式区分 early response、durable benefit、late resistance、toxicity stop；长期耐药不能反驳早期获益。',C.red],
  ['P1','匹配评分','Rerank加入药物、剂量、治疗线、耐药状态、CNS/LM、单药/联合、时间窗七维匹配。',C.amber],
  ['P1','预测与置信度解耦','partial/analog默认降低 confidence；只有方向明确且维度匹配的证据才能改变预测类别。',C.amber],
  ['P2','数据库补全','建立早期CNS结局库与毒性/停药风险库；病例卡片保存publication date和原文span。',C.teal],
 ];
 items.forEach((it,i)=>{
   const y=y0+i*1.02; tag(s,it[0],0.72,y+0.12,0.52,it[3]);
   s.addText(it[1],{x:1.48,y:y+0.06,w:2.0,h:0.3,fontSize:15,bold:true,color:C.navy,margin:0});
   s.addText(it[2],{x:3.45,y:y,w:9.1,h:0.55,fontSize:12,color:C.ink,margin:0,fit:'shrink',valign:'mid'});
   s.addShape(pptx.ShapeType.line,{x:1.45,y:y+0.72,w:11.05,h:0,line:{color:'D9E2EA',width:0.8}});
 });
 s.addShape(pptx.ShapeType.roundRect,{x:1.55,y:6.45,w:10.25,h:0.46,fill:{color:'E7F6F3'},line:{color:'B7E2DA'}});
 s.addText('目标行为：Question-only 形成先验 → case 仅局部更新对应维度 → 证据弱则降置信度，不自动降疗效。',{x:1.78,y:6.57,w:9.8,h:0.20,fontSize:12.5,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
 addSpeaker(s,'第16页｜改进优先级',`改进优先级有五项。第一，每条证据必须绑定它支持的结局维度，不能用 CNS 反证改变体部，也不能用长期耐药改变早期获益。第二，显式标记早期应答、长期耐药和毒性停药。第三，Rerank 增加药物、剂量、治疗线、耐药状态、CNS、单药联合和时间窗匹配。第四，把预测类别和置信度解耦，部分证据默认只降低置信度，除非它方向明确且维度高度匹配。第五，数据库需要分别补早期 CNS 结局和毒性停药风险。最终希望形成：模型先验先给判断，病例证据只做局部更新。`);
}

// 10 conclusion
{
 const s=pptx.addSlide('MASTER'); addTitle(s,'9｜结论：Case 有价值，但“相似”必须服务于具体推断','从检索相关性走向决策相关性');
 s.addShape(pptx.ShapeType.roundRect,{x:0.78,y:1.32,w:11.78,h:1.14,fill:{color:C.navy},line:{color:C.navy}});
 s.addText('Golden Context 证明 Generate 能保留正确答案；当前限制主要是上游证据是否直接，以及弱证据如何影响预测。',{x:1.15,y:1.65,w:11.05,h:0.48,fontSize:21,bold:true,color:C.white,align:'center',margin:0,fit:'shrink'});
 box(s,0.78,2.92,3.58,2.22,'已被证明','高度匹配的反证 case 能修正过度乐观预测。\n案例A中，症状方向被成功纠正。',C.green);
 box(s,4.88,2.92,3.58,2.22,'尚未被证明','当前三题没有任何 direct key finding。\n弱病例证据尚未稳定优于 Question-only。',C.amber);
 box(s,8.98,2.92,3.58,2.22,'下一步验证','扩充正确类型病例后，按结局维度做消融：\n先验 vs 局部证据更新 vs 完整Agent。',C.teal);
 s.addText('最终判断',{x:0.82,y:5.68,w:1.4,h:0.3,fontSize:17,bold:true,color:C.teal,margin:0});
 s.addText('不是“Case 没用”，而是当前系统把“医学相关的 Case”过早等同于“对当前预测直接有用的 Case”。',{x:2.10,y:5.61,w:10.0,h:0.50,fontSize:20,bold:true,color:C.navy,margin:0,fit:'shrink'});
 s.addText('评估依据：3题同一Generate消融、真实AnswerContext、真实Generate payload与真实检索chunk；不采用M7、Judge或Composite。',{x:0.82,y:6.50,w:11.7,h:0.26,fontSize:10,color:C.gray,align:'center',margin:0});
 addSpeaker(s,'第17页｜结论',`最后总结。Golden Context 已经证明 Generate 能够保留正确答案，所以当前主要限制在上游。高度匹配的病例证据确实有价值，第一题已经展示了它如何纠正症状方向。但三道题都没有直接 key finding，因此还不能证明当前弱病例证据链稳定优于只给问题。问题不在于 case 没用，而在于系统把“医学上相关”过早等同于“对当前具体预测直接有用”。下一步需要扩充正确类型病例，并用按结局维度的局部更新重新做消融。`);
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(SCRIPT, '# 病例证据是否帮助 Agent 答题｜逐页逐字稿\n\n' + notes.join('\n'), 'utf8');
pptx.writeFile({ fileName: OUT });
