import operator
from typing import Annotated, Literal, TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from common import get_llm, with_retry, decide, ask, dialogue_to_str, text_of
from focusing import run_focusing, experiments 

BENCH = """The bench: DMD of 64x64 pixels (binary amplitude), scattering medium, camera of 32x32 pixels.
The focus is a disk at the camera centre; the background is a ring around it, cut into sectors.
Enhancement = focus intensity / background intensity. More macropixels and more iterations give a higher enhancement."""


class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    decision_count: int
    instruction: str
    best: float | None


ORCH_SYS = f"""You are the Orchestrator. You write no code and have no tools.
Roles:
- scientist: chooses the experiment settings (a short list of runs with different settings)
- general: answers short side questions
- executor: the only role with the tool run_focusing; runs the experiments and reports the enhancements the tool returned
- finish: stop (only after the executor ran at least 3 experiments)
You have at most 8 decisions. Return next_role, an instruction, and a one-sentence reason.
{BENCH}"""

SCI_SYS = f"""You are the Scientist. You have no tools. Choose the settings for 3 to 5 focusing experiments:
macropixel_size (divides 64), focus_radius, background_radius, n_regions, n_iterations, n_flips. Vary them so we learn what helps,
and aim for a high enhancement. If results already exist, improve the settings. Give the list briefly, with a reason for each.
Never claim a number was measured unless the executor reported it.
{BENCH}"""

GEN_SYS = "You are the General Agent. Answer short side questions about the bench or the terms. No tools. Do not invent measured numbers."

EXEC_SYS = f"""You are the Executor, the only role with the tool run_focusing. Run the Scientist's latest list of experiments by calling the tool.
If the tool returns an ERROR, fix the settings and call again. Report only the enhancements the tool returned.
{BENCH}"""


def orchestrator(state: State) -> Command[Literal["scientist", "general", "executor", "__end__"]]:
    n = state["decision_count"]
    if n >= 8:
        return Command(goto=END, update={"messages": [AIMessage(content="[Orchestrator] Max 8 decisions reached -> stopping")]})
    facts = f"Decisions used: {n} of 8. Experiments run: {len(experiments)}. Best enhancement: {state['best']}."
    d = decide(ORCH_SYS, f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n{facts}\nMake your decision.")
    role, instruction, reason = d.next_role, d.instruction, d.reason
    if role == "finish" and len(experiments) < 3:
        has_plan = any(m.content.startswith("[Scientist]") for m in state["messages"])
        role = "executor" if has_plan else "scientist"
        instruction = "Run the experiments." if has_plan else "Choose the experiment settings."
        reason += " [overridden: fewer than 3 experiments run]"
    msg = AIMessage(content=f"[Orchestrator] decision {n + 1}: next={role} | instruction={instruction} | reason={reason}")
    update = {"messages": [msg], "decision_count": n + 1, "instruction": instruction}
    return Command(goto=END if role == "finish" else role, update=update)


def make_role(tag, system):
    def node(state: State) -> Command[Literal["orchestrator"]]:
        human = f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\nInstruction from the orchestrator: {state['instruction']}"
        return Command(goto="orchestrator", update={"messages": [AIMessage(content=f"[{tag}]\n{ask(system, human)}")]})
    return node


def executor(state: State) -> Command[Literal["orchestrator"]]:
    start = len(experiments)
    llm = get_llm(2048).bind_tools([run_focusing])
    human = f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\nInstruction from the orchestrator: {state['instruction']}"
    msgs = [SystemMessage(content=EXEC_SYS), HumanMessage(content=human)]
    final = ""
    for _ in range(6):
        ai = with_retry(lambda: llm.invoke(msgs))
        msgs.append(ai)
        if not ai.tool_calls:
            final = text_of(ai)
            break
        for tc in ai.tool_calls:
            print(f"[executor calls run_focusing] {tc['args']}")
            result = run_focusing.invoke(tc["args"])
            msgs.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
    log = "\n".join(f"- {e}" for e in experiments[start:]) or "- no experiment was run this turn"
    update = {"messages": [AIMessage(content=f"[Executor]\n{final}\n\n[experiments this turn]\n{log}")]}
    if experiments:
        update["best"] = max(e["enhancement"] for e in experiments)
    return Command(goto="orchestrator", update=update)


builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("scientist", make_role("Scientist", SCI_SYS))
builder.add_node("general", make_role("General", GEN_SYS))
builder.add_node("executor", executor)
builder.add_edge(START, "orchestrator")
graph = builder.compile()

init = {"messages": [HumanMessage(content="Focus the beam.")], "decision_count": 0, "instruction": "", "best": None}
for event in graph.stream(init, stream_mode="values", config={"recursion_limit": 60}):
    print(event["messages"][-1].content)
    print("-" * 60)

print("\n| run | macropixel | focus_r | bg_r | regions | iters | flips | enhancement |")
print("|---|---|---|---|---|---|---|---|")
for e in experiments:
    print(f"| {e['run']} | {e['macropixel']} | {e['focus_r']} | {e['bg_r']} | {e['regions']} | {e['iters']} | {e['flips']} | {e['enhancement']} |")