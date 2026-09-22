# DkbgParser

面向后续“文章理解与数据抽取”的文档解析底座。当前支持 Word 和文本型 PDF，输出统一、可序列化的结构与样式模型；整份扫描版 PDF 暂不支持，但文本型 PDF 中的钻孔柱状图可使用本地 OCR 补充识别。

## 当前能力

当前核心代码按职责分层：

- `parser_engine/loaders/`：Word/PDF/OpenDataLoader 文档解析；
- `parser_engine/extraction.py`：抽取流程编排与文档结构处理；
- `parser_engine/extraction_parts/config.py`：YAML 配置加载、校验、正则检查；
- `parser_engine/extraction_parts/rules.py`：派生字段规则执行；
- `parser_engine/extraction_parts/business.py`：耕土合并、逐层桩型等公共业务方法；
- `parser_engine/image_recognition.py`：图片识别流程编排和本地 RapidOCR；
- `parser_engine/image_parts/vision.py`：兼容多模态接口的视觉 HTTP 客户端；
- `parser_engine/image_parts/aggregation.py`：钻孔观测标准化、深度校验和厚度聚合；
- `parser_engine/callback/mapping.py`：业务结果到回调接口字段映射；
- `parser_engine/callback/client.py`：逆向地质 HTTP POST；
- `parser_engine/logging_utils.py`：控制台/文件日志和阶段耗时。

旧的 `parser_engine.extraction`、`parser_engine.image_recognition`、
`parser_engine.callback_mapping` 对外导入方式继续兼容，结构重构不要求调用方改代码。

- Word：`.doc/.docx/.docm/.dot/.dotx/.rtf/.odt`，使用本地 Aspose.Words
- PDF：主流程默认使用 OpenDataLoader；也可切换本地 Aspose.PDF
- 统一模型：页面、段落、标题、列表、表格、文本 span、字体样式、PDF 坐标
- 扫描件检测：大部分页面只有图片且几乎没有文本时抛出 `ScannedPdfNotSupportedError`
- 直接运行入口与 Python API，均可输出 UTF-8 JSON

## 安装与运行

项目的 `runtime/` 已包含 Aspose DLL 和许可证。安装 Python 依赖：

```powershell
python -m pip install -e .
```

在 `main.py` 文件末尾给 `main()` 传入报告路径，然后直接运行，不需要命令行参数：

```python
main(
    input_file=r"地勘报告案例\项目地勘报告.pdf",
)
```

```powershell
python main.py
```

每次运行同时输出控制台日志和报告独立日志：
`output/logs/<报告名>.log`。日志会记录文档解析与业务抽取、接口字段映射、
逆向接口 POST 三个阶段的开始、完成、耗时以及接口响应，便于定位性能和接口问题。

JSON 文件名会根据输入文件自动生成。例如：

```text
输入：地勘报告案例/项目地勘报告.pdf
输出：output/项目地勘报告.json
```

Python API：

```python
from parser_engine import DocumentParser

document = DocumentParser().parse("input.docx")
print(document.text)
for block in document.blocks:
    print(block.kind, block.page, block.text)
```

OpenDataLoader 是 PDF 专用的可选后端，擅长阅读顺序、表格和边界框；它当前不处理 Word。安装后可这样切换：

```powershell
python -m pip install -e ".[opendataloader]"
```

然后调用时传入 `pdf_backend="opendataloader"`。

也可通过环境变量覆盖运行库位置：

- `ASPOSE_DLL_DIR`：包含 `Aspose.Words.dll` 和 `Aspose.PDF.dll` 的目录
- `ASPOSE_LICENSE_PATH`：许可证文件路径

## 面向后续抽取的设计

`DocumentModel` 是稳定的中间层。建议后续按以下方向增加处理器，而不是把业务规则放进 Aspose loader：

1. 版式归一化：合并 PDF 行、去页眉页脚、恢复跨页段落。
2. 文档结构：结合 Word 标题样式或 PDF 字号/位置构建章节树。
3. 语义抽取：按章节和表格生成 chunk，再用规则、schema 或 LLM 抽字段。
4. 可追溯性：抽取结果记录 `block.id`、页码与 `bbox`，能回到原文定位。

解析层已经保留这些步骤需要的证据。因此它可以支撑“了解文章内容再抽取”，但语义层还需要根据目标字段、文档类型和准确率要求另行实现。

## 配置化抽取

`ExtractionEngine` 可以在同一份文档解析结果上执行一个或多个 YAML 配置。当前把
PRD 中的需求归纳成三种公共模式：

- `layer_records`：正文层描述、物理力学统计表、承载力推荐表按层关联；
- `keyword_fields`：优先指定章节、关键词附近取值、否定词过滤和候选值选择；
- `section_content`：按标题别名提取完整章节，并保留段落、表格、图片和页码证据。

`section_content` 中，单一来源章节使用 `aliases`；一个目标章节需要汇总多个原文
小节时使用 `alias_groups`。每个分组选择一个最准确的标题，最后按原文顺序合并。
例如“抗震地段划分”分别配置“抗震设防烈度”和“场地特征周期”两组，避免只取
第一处命中。

对应配置分别是 `configs/layer_thickness.yaml`、`configs/report_fields.yaml` 和
`configs/report_sections.yaml`。

配置加载时会校验任务模式、必填字典、章节键、`alias_groups` 二维结构以及正则表达式；
配置错误会在程序启动阶段直接给出路径明确的异常，不会静默变成“没有结果”。

### PRD 2.2.5 文字稿内容提取

配置文件为 `configs/report_sections.yaml`。目前覆盖以下 14 个目标章节：

| 目标层级 | 输出章节 | 主要识别标题或关键词 |
| --- | --- | --- |
| `{2,1,1}` | 水文气象 | 气象水文、气候气象、区域水文 |
| `{2,1,2}` | 区域地形地貌 | 区域地形地貌、地形地貌 |
| `{2,1,3}` | 区域地质构造 | 区域地质构造、构造地质条件、区域地质、区域稳定 |
| `{2,1,4}` | 地层岩性 | 地层岩性、场地地层结构及岩土物理力学性质、地层岩性分布特征 |
| `{2,1,5}` | 地下水条件 | 地下水条件、地下水特征 |
| `{3,2,1}` | 场地土的腐蚀性评价 | 场地土、地基土腐蚀性评价 |
| `{3,2,2}` | 不良地质现象及地质灾害 | 不良地质作用、特殊性岩土、地质灾害 |
| `{3,2,3}` | 岩体质量分类及岩土体物理力学参数 | 岩土物理力学性质、主要物理及工程特性指标推荐 |
| `{3,2,4}` | 场区工程地质条件评价 | 场地稳定性及适宜性评价 |
| `{3,2,5}` | 地基评价 | 地基基础方案概述、地基基础方案分析、地基评价 |
| `{3,3,1}` | 抗震地段划分 | 抗震设防烈度、场地特征周期、场地地震效应 |
| `{3,3,2}` | 地基土类型及场地类别 | 场地类别、地基均匀性、地基稳定性 |
| `{3,3,3}` | 地震作用 | 地震液化、液化判别、地震稳定性评价 |
| `{4}` | 结论 | 结论、结论与建议、结论与评价 |

执行 `python main.py` 后会生成四个 JSON：

```text
output/<输入文件名>.json          # 精简业务结果，平时读取这个
output/<输入文件名>_details.json  # 完整候选和证据，追溯查询时读取
output/<输入文件名>_fields.json   # result.json 每个key的中文含义、类型和单位
output/<输入文件名>_callback.json # 逆向更新地质数据接口请求体
```

精简文件按“土层 + 该层逐钻孔观测 + PRD 2.2.2～2.2.8 汇总结果”组织，不包含 `document`、
`tasks`、候选记录和证据等查询辅助结构。空值字段不会写入精简结果；完整明细仍位于
`_details.json` 的 `tasks.*.selected_records`。

每条文字稿章节记录包含：

- `output_title`：生成文字稿时使用的统一章节标题；
- `outline_path`：PRD 指定的目标层级，例如 `[2, 1, 1]`；
- `text`：章节合并后的纯文本；
- `status` 和 `placeholder`：仅待外部补充时输出。

`source_title`、`source_outline`、逐块 `content`、页码、表格和图片坐标只保存在明细文件。

报告缺少“水文气象”或“区域地形地貌”且按 PRD 需要百度搜索时，当前不会访问网络，
而是保留待实现占位。例如：

```json
{
  "output_title": "水文气象",
  "outline_path": [2, 1, 1],
  "text": "",
  "status": "pending",
  "placeholder": {
    "type": "external_search",
    "provider": "baidu",
    "status": "not_implemented",
    "query_template": "${project_county_or_city} 水文气象",
    "message": "报告未找到对应章节，百度搜索暂未实现"
  }
}
```

出现一个或多个 `pending` 章节时，整个 `report_sections.status` 为 `partial`，并在
`warnings` 中列出待补章节。PRD 指定了固定兜底文案的“不良地质”和“场地稳定性”章节，
报告未命中时会输出 `source: fallback`、`status: defaulted`。

直接解析文档并执行配置：

```python
import json

from parser_engine import extract_document

result = extract_document(
    "地勘报告.pdf",
    [
        "configs/layer_thickness.yaml",
        "configs/report_fields.yaml",
        "configs/report_sections.yaml",
    ],
    pdf_backend="opendataloader",
    output_path="output/地勘报告.json",
    details_output_path="output/地勘报告_details.json",
    field_descriptions_output_path="output/地勘报告_fields.json",
)

print(json.dumps(result, ensure_ascii=False, indent=2))
```

`main.py` 已默认加载全部 PRD 配置。正文没有层厚，或者识别层数少于报告原文声明数量时，
会使用免费的本地 OCR 补漏；调用 `main()` 时可通过 `enable_ocr=False` 关闭。
基础形式不需要传参。程序会逐层判断：土层名称包含“岩”或“石”时采用灌注桩参数，
其余土层采用预制桩参数。

如果有多个配置，把路径作为列表传入即可，文档不会重复解析：

```python
result = extract_document(
    "地勘报告.pdf",
    [
        "configs/layer_thickness.yaml",
        "configs/bearing_capacity.yaml",
    ],
)
```

也可以复用已经解析好的 `DocumentModel`：

```python
from parser_engine import DocumentParser, ExtractionEngine

document = DocumentParser().parse("地勘报告.docx")
engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
result = engine.extract_all(document)
```

完整明细文件中的每个任务包含：

- `records`：从原文识别到的全部候选记录；
- `selected_records`：按完整层号整理后的有效土层结果；
- `evidence`：候选值对应的原文、页码和内容块编号；
- `status` 和 `warnings`：任务状态及章节未找到等提示。

精简结果文件按需求文档保留：

- `geotechnical_layer_parameters`：PRD 2.2.2 以土层为主；每个土层的 `boreholes` 列表保存该层在多个钻孔中的厚度、层底深度和层底标高，土层外层保存统计平均值及设计参数；
- `seismic_parameters`：PRD 2.2.3 地震动峰值加速度、烈度、特征周期、场地类别和地震分组；
- `key_data`：PRD 2.2.4 水土腐蚀性、地基处理、持力层描述和地下水埋深；
- `draft_content`：PRD 2.2.5 文字稿章节，使用源报告真实的“章节编号 标题 -> 正文”映射；汇总型目标会拆成多个真实来源小节，不拼接标题，多个目标引用同一个真实章节时只保留一份正文；
- `site_geological_conditions_and_evaluation`：PRD 2.2.6 场区地质条件与评价；
- `regional_hydrology`：PRD 2.2.7 区域水文情况；
- `conclusion_and_evaluation`：PRD 2.2.8 结论与评价。

`layer_thickness` 中保留的物理力学结果为：

- `thickness_range`：原文厚度区间，包含 `min`、`max` 和单位；
- `average_thickness`：原文明确给出的平均厚度；
- `maximum_exposed_thickness`：未揭穿土层的最大揭露厚度；
- `thickness`：按照分组和优先级规则选出的报告厚度；
- `boreholes`：该土层在不同钻孔中的观测列表，每项只包含钻孔号及实际识别到的层厚、层底深度、层底标高；来源页和置信度在明细文件中；

- `gravity_density`：重力密度 γ，统一为 kN/m³；原表为 g/cm³ 或 t/m³ 时乘以 9.8；
- `cohesion`：黏聚力/内聚力 C，kPa；优先取原文平均值、其次取报告推荐值，均缺失时按土类型缺省规则赋值；
- `friction_angle`：摩擦角 Φ，度；优先取原文平均值、其次取报告推荐值，均缺失时按土类型缺省规则赋值；
- `compression_modulus_es1_2`：压缩模量 Es1-2，MPa；优先取原文平均值、其次取报告推荐值，均缺失时按土类型缺省规则赋值；
- `side_friction_fs`：物理力学统计表中的 fs 平均值，kPa；
- `clay_content`：ρc 黏粒含量，%；用于粉土相关规则判断，不作为桩端阻力；
- `pile_side_resistance`：从桩基参数表 qsik 获取，并按当前土层名称选择对应桩型列；
- `pile_tip_resistance`：从桩基参数表 qpk 获取，并按当前土层名称选择对应桩型列；
- `poisson_ratio`：泊松比；
- `bearing_capacity_fak`：承载力特征值 fak，kPa。
- `width_bearing_coefficient_eta_b`、`depth_bearing_coefficient_eta_d`：承载力宽度、深度修正系数；
- `seismic_bearing_coefficient_zeta_a`：地基抗震承载力调整系数；
- `liquefaction_reduction_coefficient`：按 N/Nr 与试验深度得到的液化折减系数；
- `cast_in_place_*`、`precast_*`：从桩基参数表保留的灌注桩/预制桩候选值；最终业务值仍按当前土层名称逐层选择；
- `horizontal_resistance_coefficient`、`negative_friction_coefficient`、`uplift_coefficient`：PRD 要求的桩基派生参数。

标准贯入/动力触探击数、含水量、孔隙比、液塑限、候选值、控制钻孔编号、参数来源、
选值过程、来源页、OCR 置信度和原文证据只保存在 `<输入文件名>_details.json`。
按土类型补值时，明细记录的 `derived_field_sources` 会标记为“土类型缺省规则”，
便于区分报告原值与规则赋值。具体数值维护在 `configs/layer_thickness.yaml` 的
`derived_fields` 中，不需要修改 Python 代码。

### 逆向更新地质数据接口映射

主程序生成 `<报告名>.json` 后，会自动调用 `write_reverse_geology_payload()`，再生成：

- `<报告名>_callback.json`：可作为
  `/rpc-api/reverse-callback/parse-reverse-geology` 的直接 JSON 请求体；
- `<报告名>.json`：原有精简业务结果，内容不变；
- `<报告名>_details.json`：完整查询明细，内容不变。

运行其他报告时不需要另外编写映射代码，只需给 `main()` 传入需要的参数：

```python
main(
    input_file=r"地勘报告案例\项目地勘报告.pdf",
    project_id=123,
    geology_id=0,
    layer_ids={"②-1": 456},
    handle_keyword_codes={"岩溶": 1, "湿陷性黄土": 2},
)
```

主流程生成映射 JSON 后，会默认把该 JSON 直接 POST 到
`http://172.16.14.71:10004/rpc-api/reverse-callback/parse-reverse-geology`。
可通过 `main(callback_url=...)` 覆盖地址，`callback_timeout` 调整超时，
本地只生成文件时可传 `send_callback=False`。

完整映射关系集中在 `parser_engine/callback/mapping.py` 的
`CALLBACK_FIELD_MAPPING` 和 `CALLBACK_LAYER_FIELD_MAPPING` 两个字典中；
HTTP POST 位于 `parser_engine/callback/client.py`。旧的
`parser_engine/callback_mapping.py` 只作为兼容导入入口保留。映射字典每一项都有
中文注释，键是接口字段，值是 result.json 来源字段。

接口中的 `projectId`、`geologyId` 和岩土层 `id` 属于业务数据库标识，需要调用方
在 `main.py` 配置。桩侧阻力、桩端阻力、水平抗力比例系数和负摩擦阻力系数均使用
同一个公共方法逐层选择：层名含“岩”或“石”选灌注桩值，其余选预制桩值。
`handleKeyword` 的编码未在接口文档中给出，因此有处理关键字时必须显式提供
`handle_keyword_codes`。若报告结果为“中强腐蚀性”，接口却要求区分中腐蚀 `2`
和强腐蚀 `3`，需传入 `ambiguous_corrosion_code=2` 或 `3`。

callback JSON 默认保留完整接口字段；没有抽取到或没有映射上的字段返回 `null`，
不会用伪造的 `0` 代替。未配置的含混枚举同样返回 `null`，并在主程序日志中提醒。

土层业务结果还有一项统一预处理规则：如果第一层名称包含“耕土”且存在下一层，
抽取引擎会先删除该耕土层，并把耕土厚度累加到下一层；此后的派生参数、最后一层处理、
`result.json` 和 `callback.json` 全部基于合并后的土层列表。`details.json` 的
`records` 仍保留原始识别候选，`selected_records` 则保存合并后的业务结果。

最后一层“识别厚度 +20m”属于后续工程计算调整：精简结果中的 `thickness` 输出调整后
厚度；调整前值、增加值和调整后值同时保存在明细 JSON 的 `effective_value`、
`adjustment` 和 `final_value` 中。若最终首层为“耕土”且存在下一层，业务结果会删除
该耕土层并把其厚度合并到下一层；原始识别候选仍保留在 details.json 的 `records` 中，
便于追溯。

### 钻孔柱状图识别

图片兜底会定位低文本附图页和钻孔柱状图页，逐页识别多个钻孔，再按完整层号汇总
有效厚度；具体使用 average/minimum/maximum 由 YAML 的聚合配置决定，
`depth_validation=False` 的观测保留在明细中但不参与统计。随后再与正文物理力学
统计表、推荐值表按层号合并。页面使用 PyMuPDF
渲染为高清 PNG，再调用注入的识别客户端。安装图片渲染依赖：

```powershell
python -m pip install -e ".[opendataloader,image]"
```

免费、离线的本地 OCR 使用 RapidOCR，不需要接口地址、账号或 API Key：

```powershell
python -m pip install -e ".[opendataloader,ocr]"
```

```python
from parser_engine import BoreholeImageRecognizer, RapidOCRClient, extract_document

recognizer = BoreholeImageRecognizer(
    RapidOCRClient(),
    output_dir="output/assets",
)

result = extract_document(
    "地勘报告.pdf",
    "configs/layer_thickness.yaml",
    pdf_backend="opendataloader",
    output_path="output/layer_thickness.json",
    image_recognizer=recognizer,
)
```

本地 OCR 适合表格版式较固定的钻孔柱状图。层号下标、复杂合并单元格等小字可能误识别，
因此 JSON 会保存原始页码、渲染图片路径和 OCR 坐标，正式使用前应抽样复核。
识别阈值、默认列坐标比例和岩土名称正则位于
`configs/layer_thickness.yaml` 的 `image_fallback.ocr`，可按报告版式覆盖。

连接兼容多模态 Chat Completions 格式的本地或云端服务：

```python
from parser_engine import (
    BoreholeImageRecognizer,
    OpenAICompatibleVisionClient,
    extract_document,
)

client = OpenAICompatibleVisionClient(
    api_url="http://127.0.0.1:8000/v1/chat/completions",
    model="your-vision-model",
    # 云端服务可以传 api_key，或设置 VISION_API_KEY 环境变量。
)
recognizer = BoreholeImageRecognizer(
    client,
    output_dir="output/assets",
)

result = extract_document(
    "地勘报告.pdf",
    "configs/layer_thickness.yaml",
    pdf_backend="opendataloader",
    output_path="output/layer_thickness.json",
    image_recognizer=recognizer,
)
```

只有显式传入 `image_recognizer` 才会发送图片。渲染页面会保存在 `output/assets/<文档名>/`，抽取结果通过 `image_path`、页码和钻孔编号保留证据。对于无法通过关键词定位的附图，可在 YAML 的 `image_fallback.pages` 中直接填写页码；否则使用 `fallback_last_pages` 检查文档末尾页面。
