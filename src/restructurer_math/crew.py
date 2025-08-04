from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Configure Ollama-compatible LLM
# llm = LLM(
#     model="ollama/llama3.2",
#     base_url="https://two-jeans-ring.loca.lt",
#     temperature=0.1,
#     max_tokens=2000
# )

@CrewBase
class Restructure():
    """Restructure crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def question_restructure_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['question_restructure_agent'], # type: ignore[index]
            verbose=True,
        )

    @agent
    def marking_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['marking_agent'], # type: ignore[index]
            verbose=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def restructure_answers_task(self) -> Task:
        return Task(
            config=self.tasks_config['restructure_answers_task'], # type: ignore[index]
        )

    @task
    def mark_allocation_task(self) -> Task:
        return Task(
            config=self.tasks_config['mark_allocation_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Restructure crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )