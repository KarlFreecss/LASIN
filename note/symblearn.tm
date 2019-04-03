<TeXmacs|1.99.4>

<style|article>

<\body>
  <doc-data|<doc-title|Statistical Learning by Logical
  Abduction>|||||||<doc-author|<author-data|<author-name|daiwz>|<\author-affiliation>
    LAMDA Group

    November End, 2015
  </author-affiliation>>>>

  <section|Problem Setting>

  <strong|Input:>

  <\itemize>
    <item>Training data: <math|<around*|(|\<b-x\><rsub|i>,\<b-y\><rsub|i>|)>>,
    <math|x> is feature, <math|\<b-y\>=<around*|(|y<rsub|1>,\<ldots\>,y<rsub|l>|)>>
    is a vector of multiple labels.

    <item>Background knowledge:\ 

    <\itemize>
      <item>Label relations: e.g. label hierarchy
      (<verbatim|father(label_i,label_j)>,...).

      <item>Other knowledge: template (meta-rules) for rule abduction;
      knowledge of constructing meta-rules. For example,
      <verbatim|father(P,Q)><math|\<rightarrow\>><verbatim|metarule(Q(X):-P(X),R(X))>,
      meaning if label <verbatim|P> is father of label <verbatim|Q>, then
      every rule about <verbatim|Q> must include <verbatim|P> in its body.
    </itemize>
  </itemize>

  <no-indent><strong|Output:>

  <\itemize>
    <item>A set of rules as hypothesis of each label.

    <item>Predicates in rules are statsitical classifiers, i.e.,
    <verbatim|P(X):-(><math|h<rsub|p>>(<verbatim|X>)<verbatim|\<gtr\>0).>
  </itemize>

  <\strong>
    Main Idea:
  </strong>

  <with|color|blue|Background knowledge <math|\<Longrightarrow\><rsup|<small|<tiny|\<spadesuit\>>>>>
  Meta-rules (template rules) <math|\<Longrightarrow\>> Statistical Models
  learning.>

  <small|<yes-indent><space|1em><math|<rsup|<tiny|\<spadesuit\>>>\<Longrightarrow\>>
  means \Pcontrol\Q>

  <\itemize>
    <item>Incorporating first-order background knowledge by letting them to
    control the learning of statistical classifiers. The meta-rules act as
    templates and enables predicate invention, and the invented
    \Ppredicates\Q are statistical classifiers.\ 

    <item>Statistical learning is directly controled by its training data,
    which is actually controlled by rule induction by variable bindings. For
    example, when learning a classifier <verbatim|R(X)> inside of
    \ \P<verbatim|tree(X):-plant(X),R(X)>\Q, the training data passed to
    <verbatim|R> is filtered by <verbatim|plant(X)>, which indicates
    <verbatim|R(X)> should only learn the difference between <verbatim|tree>
    and other <verbatim|plants>.\ 

    <item>Reuse the learned classifiers as logical predicates. The invented
    predicate (for exapmle the <verbatim|R> in last paragraph) might be an
    useful mid-level concept that can help the learning of other labels.
  </itemize>

  <section|Motivation>

  Statistical learning solves a zero-order problem \U learn a first-order
  representation of <with|color|red|target concept> (predicate/symbol) from
  low-level feature space. By assuming i.i.d. among the domain and even
  conditional independence between features (e.g.
  <math|P<around*|(|x<rsub|i>,x<rsub|j><around*|\||y|\<nobracket\>>|)>=P<around*|(|x<rsub|i><around*|\||y|\<nobracket\>>|)>P<around*|(|x<rsub|j><around*|\||y|\<nobracket\>>|)>>),
  the problem has been simplified. Moreover, discriminative statistical
  learning only tries to learn the difference between different concepts
  rather than the constructive structure (relations) of concepts, which makes
  current mainstream statistics based machine learning has a huge necessity
  of examples. In short, both of the modeling process and the learning target
  ignores lots of relations in domain and tries to solve an over-simplified
  problem.

  In fact, plenty of first-order relations between
  <with|color|red|sub-concepts> play important roles in our real cognition
  and learning problems. The more important thing is, <strong|learning is a
  constructive process, human always keep trying to interprete
  <with|color|red|new concepts (symbols)> with their <with|color|red|old
  knowledge (background knowledge)>.> \ We can look at an interesting
  example: when a child is trying to understand what is \Ptree\Q, should
  he/she understand what is \Pleaf\Q or what is \Pbranch\Q in former? In
  fact, people can recognise trees far before understanding the sub-concpets
  of tree like branches, trunks, leaves, roots and so on. They only know
  colors, lights, edges between objects or \ other relatively lower-level
  features. After knowing what is a tree, they will continuously learn the
  concept of branch, and be able to recognise a tree even when it losts all
  of its leaves, or been cut off and layed down on the ground as a \Pdead
  tree\Q. This example gives us a hint that the existence of background
  knowledge of <with|color|red|sub-concepts> in learning may be not so
  important, they could be <strong|abduced> by higher-level concepts.

  The next problem is how can we abduce the <with|color|red|sub-concepts>?
  Where do they come from? Here is an other observation: Human can learn
  multiple concepts at one time, clearly, by <with|color|red|telling
  differences and similarities between concepts and objects>. We believe that
  the <strong|frequently appeared similar/different parts> are the
  <with|color|red|sub-concepts>. (From this angle of view, discriminative
  learning does make a certain degree of sense, however it ignores the
  first-order relations between the differences.) However, in the learning
  process, it is hard to mining the sub-parts as frequent itemsets (it
  requires exponentially enumerations). Thus, we try to use label hierarchy
  to guide the abduction of sub-concept symbols, the difference/similarity
  between hierarchical labels are relatively more easier to capture.

  <section|Proposed Approach>

  <subsection|Meta-rules>

  Given background knowledge of label hierarchy, we can generate meta-rules
  to perform the abduction.\ 

  <\center>
    <with|gr-mode|<tuple|edit|line>|gr-frame|<tuple|scale|1cm|<tuple|0.699999gw|0.4gh>>|gr-geometry|<tuple|geometry|0.226697par|0.113352par|center>|<graphics||<text-at|plant|<point|-1.21947|0.788084>>|<text-at|tree|<point|-1.97552586320942|-0.421600079375579>>|<text-at|grass|<point|-0.413017594919963|-0.396398333112846>>|<line|<point|-1.65546|-0.0560259>|<point|-0.787389204921286|0.584948405873793>|<point|0.00519248577854214|-0.102080301627199>>>>

    Fig. 1 Label hierarchy
  </center>

  For example in Fig. 1, the label hierarchy is <verbatim|father(plant,tree)>
  and <verbatim|father(plant,grass)>. Since both <verbatim|tree> and
  <verbatim|grass> are children of <verbatim|tree>, they must share some
  information, the meta-rule of these two concept should be:

  <\quote-env>
    <strong|Rule A>: <verbatim|tree(X):-plant(X),tree_1(X)>.

    <strong|Rule B>: <verbatim|grass(X):-plant(X),grass_1(X)>.
  </quote-env>

  <no-indent>The <verbatim|plant(X)> in their bodies shows the
  <with|color|red|similarity> between these two concepts, the
  <verbatim|***_1(X)> predicates are the difference between them.

  The <strong|meta-rules> are templates of first-order logic rules expressed
  by predicates in <em|list forms>. In <em|list form>, each predicate is
  expressed by a list, the first item is the name of the predicate
  (first-order), the rests are arguments (zero-order). For example, the
  <strong|rule A> can be written in <em|list form> as
  \P<verbatim|[tree,X]:-[plant,X],[tree_1,X]>\Q. By substituting all the
  constants by variables (here we use <verbatim|P,Q,R,...> to represent
  predicate variables, <verbatim|X,Y,Z,...> to represent object variables),
  the <strong|meta-rule> of these two rule are:

  <\quote-env>
    <strong|Meta Rule A>: <verbatim|[P,X]:-[Q,X],[R,X]>.
  </quote-env>

  <no-indent>If we want to enforce the learned rule to contain information
  about <verbatim|plant/2>, we just modify the meta-rule as:

  <\quote-env>
    <strong|Meta Rule B>: <verbatim|[P,X]:-[plant,X],[R,X]>.
  </quote-env>

  There could be numbers of meta-rules, according to [Muggleton et.al.,
  20xx], few of them is already enough for learning a universal Turing
  machine.

  <subsection|Abduction>

  In this section, we explain the implementation of the abduction process.

  Suppose we have already obtained label hierarchy like Fig. 1, we have
  following instructions for meta-rule construction:

  <\enumerate-roman>
    <item>Body of meta-rule of child concepts must contain its parent
    concept. i.e., <verbatim|child(X):-parent(X),...> This rule indicates the
    <verbatim|is-a> relation between concepts.

    <item><with|color|blue|[More details to be added]>...
  </enumerate-roman>

  <subsubsection|Searching meta-rules>

  Given a label to be learned, generate candidate programs. For example, if
  we want to learn <verbatim|tree(X)>, and given meta-rules:
  <verbatim|[P,X]:-[Q,X],[R,X]> and <verbatim|[P,X]:-[Q,X],[R,X],[S,X]>. The
  candidate programs are:

  <\quote-env>
    <verbatim|ps(metarule([P,X]:-[Q,X],[R,X]),metasub([P,tree],[Q,plant])).>

    <verbatim|ps(metarule([P,X]:-[Q,X],[R,X],[S,X]),metasub([P,tree],[Q,plant])).>
  </quote-env>

  <no-indent>After generating candidate programs, the learning procedure will
  start substuting <verbatim|Q,S> and assessing the quality of different
  substitution. For example, if there already exists rules of
  <verbatim|grass/1>, then there are multiple choices to continue this
  search:

  <\enumerate-numeric>
    <item>a possible <verbatim|metasub> will be
    <verbatim|[P,tree],[Q,plant],[R,grass]>;

    <item>another choice is to generate a new predicate, which results in
    <verbatim|metasub> as <verbatim|[P,tree],[Q,plant],[R,tree_1]>,the new
    predicate can be invented by:

    <\enumerate-alpha>
      <item> <with|color|red|learning a new rule>: (e.g.
      <verbatim|[tree_1,X]:-[Q,X],[R,X]>);

      <item>or simply <with|color|red|learn a statistical classifier> (e.g.
      <math|tree<around*|(|x|)>=sgn<around*|(|h<around*|(|x|)>|)>>).
    </enumerate-alpha>
  </enumerate-numeric>

  \ \ \ \ 

  <subsubsection|Training statistical models>

  The meta-rules in abduction also act as <with|color|red|training data
  filters> that selects training data for statistical model learning.

  Moreover, when learning statistical models, the biases can be flexibly
  declared in first-order logical background knowledge. For example, if the
  models to be learned are decision trees, and the background knowledge holds
  the belief that the invented mid-level concepts shouldn't be very complex,
  then the parameter of shallow depth can be passed to the \Pstatistical
  classifier invention\Q process. If the models are SVMs, we can even declare
  sparsity or other regularization in logical form and pass them to
  statistical learning process.

  <section|Implementation>

  <subsection|Out line of the system>

  The <em|Symblearn> system consist of two major parts: <verbatim|c++ data
  handler> and <verbatim|prolog abductor>. The <verbatim|c++> part takes
  charge of data processing and memory handling; the <verbatim|prolog> is
  used for rule abduction.

  <\itemize-arrow>
    <item><strong|core/>: <with|color|blue|[implementing]> directory of main
    functionalities
  </itemize-arrow>

  <\quote-env>
    <verbatim|symblearn.cpp>: <with|color|blue|[implementing]> The main
    program, initialize Prolog engine, read input data, organise data into
    prolog-readable formats.

    <verbatim|data_set.hpp>: <with|color|green|[done]> MLDataSet interface,
    in charges of data storage and access.

    <verbatim|data_patch.hpp>: <with|color|blue|[implementing]> Data patch is
    the batch of data send to prolog classifiers.

    <verbatim|classifier.cpp>: <with|color|green|[done]> C++-implemented
    logical predicates, reads data and classifier and output TRUE or FALSE
    etc. The predicates it contains are listed as follow:

    <\quote-env>
      <\itemize>
        <item><verbatim|classify_rv\|ui_set\|inst(&model,&data,Class)>: input
        <verbatim|&model> and <verbatim|&data>, output <verbatim|class> of
        data. Depend on different classifier, the output could be
        <verbatim|unsinged int> or <verbatim|RealVector>; If <verbatim|&data>
        is an adress of dataset then <verbatim|class> is a list of label
        (e.g. <verbatim|[1,0,1,1,...]>) , if <verbatim|&data> is a instance
        (<verbatim|RealVector>) then <verbatim|class> is a real value (-1 or
        1).

        <item><verbatim|filter_data(&dataset,Selector,&outputData)>: select a
        subset of data according to a binary selector
        <verbatim|[1,0,0,1,...]>.

        <item><verbatim|stat_classifier(&data,&model,Eval)>: train/evaluate a
        statistical classifier <verbatim|&model> with <verbatim|&data>;
        <verbatim|Eval> is the evaluation of <verbatim|&model> on
        <verbatim|&data>, it could be accuracy, F1 score, etc. (For
        simplicity, now we are just using fixed depth <with|color|red|CART
        Tree> implemented by Shark toolbox. [more to come]).

        <item><verbatim|free_ptr(ptr)>: free the memory address of
        <verbatim|ptr> in c-stack (not Prolog stack)

        <item><verbatim|...>
      </itemize>
    </quote-env>
  </quote-env>

  <\itemize-arrow>
    <item><strong|arff/>: <with|color|green|[done]> directory of reading
    benchmark HMC arff data(https://dtai.cs.kuleuven.be/ clus/hmc-ens/)

    <verbatim|arff_data.h>: interface of ARFF data (handles hierarchical
    labels).

    <verbatim|arff_parser.h>: parses ARFF files.

    ...

    <item><with|color|blue|[More to add]>
  </itemize-arrow>

  <subsection|Algorithm>

  <strong|Step 1>: <em|Reading data>. This job is done by <verbatim|c++>
  part. The label hierarchy or other background knowledge are also inputed in
  this stage. For example, <with|color|red|label hierarchy are parsed into
  <verbatim|father(L1,L2)>>, and asserted in the prolog engine.

  <with|color|blue|[More to add]>

  <no-indent><strong|Step X>: <em|Abduction>.

  <verbatim|Program> is a 4-tuple: <verbatim|ps(MetaSubs,Signature,SizeBound,MetaRules).>

  <verbatim|prove(Data_patch,Prog1,Prog2):->

  \;

  <section|Experiments>

  <subsection|Datasets>

  Hierarchical Multi-label datasets from Clus-HMC
  (https://dtai.cs.kuleuven.be/clus/hmc-ens/).

  <subsection|Compared Methods>

  <with|color|blue|[More to add]>
</body>

<\initial>
  <\collection>
    <associate|font-base-size|10>
    <associate|page-medium|paper>
    <associate|page-orientation|portrait>
    <associate|page-type|a4>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|4.2|4>>
    <associate|auto-11|<tuple|5|4>>
    <associate|auto-12|<tuple|5.1|4>>
    <associate|auto-13|<tuple|5.2|4>>
    <associate|auto-14|<tuple|5.3|?>>
    <associate|auto-15|<tuple|5|?>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|3|2>>
    <associate|auto-4|<tuple|3.1|2>>
    <associate|auto-5|<tuple|3.2|3>>
    <associate|auto-6|<tuple|3.2.1|3>>
    <associate|auto-7|<tuple|3.2.2|3>>
    <associate|auto-8|<tuple|4|3>>
    <associate|auto-9|<tuple|4.1|3>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Problem
      Setting> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Motivation>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Proposed
      Approach> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1<space|2spc>Meta-rules
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|3.2<space|2spc>Abduction
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|2tab>|3.2.1<space|2spc>Searching meta-rules
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|2tab>|3.2.2<space|2spc>Training statistical
      models <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Implementation>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>

      <with|par-left|<quote|1tab>|4.1<space|2spc>Out line of the system
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>

      <with|par-left|<quote|1tab>|4.2<space|2spc>Algorithm
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Experiments>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11><vspace|0.5fn>

      <with|par-left|<quote|1tab>|5.1<space|2spc>Datasets
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12>>

      <with|par-left|<quote|1tab>|5.2<space|2spc>Compared Methods
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13>>
    </associate>
  </collection>
</auxiliary>