# How to Use This Book {#preface}

This book is designed for active self-study. Read with a Python prompt, a
notebook, or a robot simulator nearby. When an equation appears, first name
every symbol and predict how changing it would affect an experiment. Then work
the numerical example before looking at the answer. When code appears, run it,
change one input, and explain the output before continuing.

The material has three connected layers:

1. **Foundations** explain the reinforcement-learning problem, the mathematics,
   and the major algorithm families.
2. **The Microduck laboratory** shows where those abstractions live in a real
   GPU-parallel robot-learning project, from environment configuration through
   ONNX deployment.
3. **Modern robot intelligence** connects locomotion to demonstrations,
   offline data, world models, vision-language-action policies, hierarchical
   planning, and reproducible research.

Microduck is the running hands-on case study, not the boundary of the subject.
Its main walking policy is a command-conditioned local controller: it does not
see a camera or obstacle map and it does not choose a destination. That clear
boundary makes it a useful base for learning how perception and planning can be
added without placing cloud latency inside a real-time balance loop.

On GitHub, chapter-end answers are collapsed so that you can attempt each
exercise first. In this PDF edition they are expanded and placed at the end of
their chapter. The primary-source index at the end records the papers and
official open-source projects behind the research chapters. Treat every
state-of-the-art claim as scoped to its dated task, data, embodiment, and
evaluation protocol.

## Suggested routes

- **Theory first:** Chapters 1–7, the four small programs in `examples/`, and
  the worked problems in Chapter 20.
- **Build and deploy:** Chapters 1–13, alongside a pinned Microduck checkout.
- **Research literacy:** Chapters 14–20 after the foundations, recording the
  evidence boundary of every paper before accepting its headline result.

Do not measure progress by pages read. Advance when you can produce the
deliverable at the end of a chapter and explain what observation, action,
reward, data, and safety contract a robot-learning result actually used.

