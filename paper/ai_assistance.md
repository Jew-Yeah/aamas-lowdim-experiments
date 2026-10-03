# AI assistance in experimental methodology and implementation

OpenAI Codex was used interactively to suggest or refine experimental scenarios,
validation and comparison plans, numerical diagnostics, and figure selection;
to implement and debug the experiment and analysis code; and to draft and
translate experimental descriptions and captions. This assistance therefore
extended to methodology and was not limited to language polishing.

The available record identifies Codex and the GPT-6 model family. Exact model
identifiers/versions across the earlier interactions and the complete prompt
history were not retained in this artifact. They have not been reconstructed
or guessed. The following are verbatim relevant user prompts available in the
retained conversation; they are an incomplete record of a longer interaction.
The original Russian wording is kept to avoid presenting a reconstruction
as an exact prompt. The English descriptions identify the purpose of each.

1. **A common application and comparison data.**
   > Предложи тогда типовой набор данных, на которых можно затестить наш алгоритм. В идеале, хотелось бы сравнить работу нашего алгоритма по схожим статьям, на которых запускался этот алгоритм.

2. **An interpretable applied task.**
   > А есть более прикладная задача? Типа хочется более понятной интепретируемости

3. **A changing opponent.**
   > Окей, просто запомним это. Давай тогда посмотрим какие-нибудь постановки, где в начале противник тупой, а потом умнеет - может там будет полезен. Как думаешь, на каких данных мы реально выигрыш получим?

4. **Implementation and experiments.**
   > Давай сразу в тестовые сценарии на реальных задачах зайдем. Напиши код и потестируй нашу идею. Вдруг что-то полезное будет.

5. **Additional statistical comparison.**
   > Короче. Потестируй еще тщательнее. Интересно, чисто статистически лучше наш метод чем CAGE или нет. Проведем больше экспериментов.

6. **Parameter selection.**
   > А что за оконная стратегия? И можем ли мы как-то подобрать параметры для нашего алгоритма, чтобы победить? Например, для бюджета ошибок

7. **Exploratory restarts.**
   > Давай посмотрим это - нам же не придется менять содержание статьи? Также есть мысль - также запускать алгоритм one-switch - ограниченно по блокам - типа у нас блок на 1000 операций отработал. Потом перезапустили с теми же данными (то есть снова продолжаем с обнуленным бюджетом ошибок) - и так далее - по блокам.
   > Что думаешь?

8. **Additional visualization.**
   > Мне кажется мало графиков... чтобы ты взял помимо средних значений и диапазона? Я бы добавил еще порядок убывания ошибки и графики соответствующие. Предложи свои варианты, какие графики наиболее полезные и информативные были бы

9. **Figure construction.**
   > Окей. давай, построй графики

10. **AAMAS presentation.**
    > Окей. Давай теперь подумаем, что закинуть в статью для AAMAS + GitHub - предварительно переведи картинки и прочее на английский язык и понятно подпиши, чтобы было понятно какие графики (что там именно разница идет). И подбери, что лучшее в саму статью на AAMAS закинуть.
    > НО! Предварительно посмотри как эксперименты описывать под AAMAS и именно в соответствии с требованиями оформи наши результаты.

The supplied protocols record the final simulator scenarios, fixed calibration,
validation selection, primary contrasts and subsequent descriptive diagnostics.
No parameter was retuned after the locked final test. Exploratory alternatives
are distinct from the selected configuration. Programmatic Matplotlib figures
display recorded numerical results; no generative image model was used to
create empirical evidence.

Automated checks cover causal action ordering, mathematical target construction,
native reward decomposition, paired bootstrap sampling units, source hashes,
and public replay. Authors remain responsible for verifying the methods,
citations, interpretation and this disclosure before submission. This statement
does not claim that automated tests replace author review, or describe AI
assistance to the theoretical manuscript beyond the experimental work recorded
here. If additional assistance affected the theory, the authors should extend
this statement using their retained records.

This record follows the [AAMAS 2027 AI policy](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/)
and [Q&A on unavailable records and multi-turn interactions](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/qa/).

## Added complete switching studies (4 October 2026)

The author requested an actual budget crossing and the entire subsequent safe
continuation, with dynamic plots and a revised experimental description. The
available instruction is retained verbatim:

> Блин, это серьезный косяк на самом деле. Давай перезапустим и описание эксперимента как раз возьмем весь процесс вместе с переключением. Графики именно отражающие динамику как сейчас в статье. Сделай это. Действительно процесс переключения имеет значение. Нет переключения - нет смысла статьи (текущий эксперимент как будто больше будет относится к исходной Marinov)

Under that direction, Codex helped identify why the frozen three-mode CAGE
paths cannot cross the original block budget, proposed the added tracking game
and constructed geometry diagnostic, derived the lag-base telescope and the
closed-form stress oracles, implemented and tested the causal runs, recovered
the public aggregate inputs from the recorded cache, and drafted the protocols,
figures, and experimental text. This includes assistance with experimental
design and mathematical implementation, in addition to search and programming.
The authors retain scientific responsibility and the final decision on use.

These post-review additions are exploratory. The NYC tracking master uses a
separately certified conservative lag-base budget; the constructed diagnostic
keeps the original block routine and budget. Neither is represented as the
earlier locked CAGE test or evidence for the q=4 rate. All five controls and
mixed outcomes are reported. No confidence intervals or p-values are attached
to the fixed traces. The theoretical statements of the manuscript were not
changed in this revision. The model-version and incomplete-history limitations
above still apply.
