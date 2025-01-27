from typing import Dict, Any, List, Optional, Union, Generator

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

# Import local modules for retrieval and generation
from retrieval.retrieval_system import RetrievalSystem
from generation.generate_response import ResponseGenerator
from agents.ice_breaker_agent import IceBreakerAgent

class IceBreakerTool(BaseTool):
    """
    Tool wrapper for the IceBreakerAgent to be used by LLM
    """
    name: str = "ice_breaker_detector"
    description: str = """
    Detects and handles common greeting and ice breaker phrases in multiple languages.
    Use this tool to check if an input is a greeting or casual conversation starter.
    Returns a friendly response if the input is an ice breaker.
    """

    def __init__(self, ice_breaker_agent):
        """
        Initialize the tool with an ice breaker agent
        
        Args:
            ice_breaker_agent: Instance of IceBreakerAgent
        """
        # Use a dictionary to store the agent to avoid Pydantic field issues
        super().__init__()
        self._agent = ice_breaker_agent

    def _run(self, input_text: str) -> str:
        """
        Run the ice breaker detection
        
        Args:
            input_text (str): Input text to check
        
        Returns:
            str: Ice breaker response or empty string
        """
        result = self._agent.process(input_text)
        return result['response'] if result['is_ice_breaker'] else ""

    async def _arun(self, input_text: str) -> str:
        """
        Async version of _run method
        
        Args:
            input_text (str): Input text to check
        
        Returns:
            str: Ice breaker response or empty string
        """
        return self._run(input_text)

class ConversationWorkflow:
    """
    LangGraph-based workflow for managing conversation and retrieval
    """
    def __init__(self, 
                 retrieval_system: RetrievalSystem, 
                 response_generator: ResponseGenerator,
                 ice_breaker_agent: Optional[IceBreakerAgent] = None,
                 enable_ice_breaker: bool = True):
        """
        Initialize the workflow graph
        
        Args:
            retrieval_system: System for retrieving relevant context
            response_generator: System for generating responses
            ice_breaker_agent: Optional custom ice breaker agent
            enable_ice_breaker: Flag to enable/disable ice breaker functionality
        """
        self.retrieval_system = retrieval_system
        self.response_generator = response_generator
        
        # Initialize ice breaker agent and respect enable flag
        self.ice_breaker_agent = ice_breaker_agent or IceBreakerAgent()
        self.enable_ice_breaker = enable_ice_breaker and ice_breaker_agent is not None
        
        # Create ice breaker tool
        self.ice_breaker_tool = IceBreakerTool(self.ice_breaker_agent)
        
        self.workflow_result = None  # Store workflow result for later access
        
        # Initialize LLM with ice breaker tool
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.7,
            model_kwargs={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": self.ice_breaker_tool.name,
                            "description": self.ice_breaker_tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "input_text": {
                                        "type": "string",
                                        "description": "The input text to check for ice breaker phrases"
                                    }
                                },
                                "required": ["input_text"]
                            }
                        }
                    }
                ]
            }
        )
        
        # Create the workflow graph with updated configuration
        self.graph = self.create_workflow_graph()

    def create_workflow_graph(self) -> StateGraph:
        """
        Create the state graph for the conversation workflow
        
        Returns:
            Configured StateGraph
        """
        class ConversationState(dict):
            input: str
            chat_history: List[Union[HumanMessage, AIMessage]]
            context: Optional[str]
            response: Optional[str]
            retrieved_texts: Optional[List[str]]
            is_ice_breaker: bool
            error: Optional[str]

        graph = StateGraph(ConversationState)

        # Define nodes for different stages of conversation processing
        if self.enable_ice_breaker:
            graph.add_node("check_ice_breaker", self.check_ice_breaker)
            graph.set_entry_point("check_ice_breaker")
            
            # Conditional routing based on ice breaker check
            graph.add_conditional_edges(
                "check_ice_breaker",
                self.route_after_ice_breaker,
                {
                    "ice_breaker": END,  # Stop if it's an ice breaker
                    "continue": "retrieve_context"  # Continue to normal workflow
                }
            )
        else:
            # If ice breaker is disabled, start directly with retrieve_context
            graph.set_entry_point("retrieve_context")

        # Always add these nodes
        graph.add_node("retrieve_context", self.retrieve_context)
        graph.add_node("generate_response", self.generate_response)
        graph.add_node("handle_error", self.handle_error)

        # Connect nodes for normal workflow
        if not self.enable_ice_breaker:
            graph.add_edge("retrieve_context", "generate_response")
        
        graph.add_conditional_edges(
            "generate_response",
            self.route_response,
            {
                "success": END,
                "error": "handle_error"
            }
        )
        graph.add_edge("handle_error", END)

        return graph.compile()

    def check_ice_breaker(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the input is an ice breaker
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with ice breaker check result
        """
        # Ensure we're only checking the current input
        current_input = state.get('input', '')
        
        try:
            ice_breaker_result = self.ice_breaker_agent.process(current_input)
            return {
                **state, 
                'is_ice_breaker': bool(ice_breaker_result['is_ice_breaker']),
                'response': ice_breaker_result['response'] if ice_breaker_result['is_ice_breaker'] else None,
                'retrieved_texts': []  # Ensure retrieved_texts is an empty list for ice breakers
            }
        except Exception as e:
            return {
                **state, 
                'error': str(e), 
                'is_ice_breaker': False,
                'retrieved_texts': []
            }

    def route_after_ice_breaker(self, state: Dict[str, Any]) -> str:
        """
        Determine next step after ice breaker check
        
        Args:
            state: Current conversation state
        
        Returns:
            Next node to execute
        """
        return 'ice_breaker' if state.get('is_ice_breaker') else 'continue'

    def retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant context for the input
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with retrieved context
        """
        try:
            # Use retrieval system to find relevant context
            retrieved_texts = self.retrieval_system.retrieve(state['input'])
            return {**state, 'retrieved_texts': retrieved_texts}
        except Exception as e:
            return {**state, 'error': str(e)}

    def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response using the context and input
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with generated response
        """
        try:
            # Use response generator with retrieved context
            response_generator = self.response_generator.generate(
                state.get('retrieved_texts', []),
                state['input'],
                previous_context=state.get('chat_history', []),
                llm=self.llm
            )
            
            # Collect the full response (for streaming-compatible workflows)
            response_text = ''.join(list(response_generator))
            
            return {**state, 'response': response_text}
        except Exception as e:
            return {**state, 'error': str(e)}

    def route_response(self, state: Dict[str, Any]) -> str:
        """
        Determine the next step based on the state
        
        Args:
            state: Current conversation state
        
        Returns:
            Next node to execute
        """
        return 'success' if state.get('response') else 'error'

    def handle_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle any errors that occur during processing
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with error message
        """
        error_msg = state.get('error', 'An unknown error occurred')
        return {**state, 'response': f"Error: {error_msg}"}

    def run(self, 
            input_text: str, 
            chat_history: List[Union[HumanMessage, AIMessage]] = None
    ) -> Generator[str, None, None]:
        """
        Run the workflow graph with the given input
        
        Args:
            input_text: User input text
            chat_history: Optional chat history
        
        Returns:
            Generator of response tokens or full workflow result
        """
        # Prepare initial state
        initial_state = {
            'input': input_text,
            'chat_history': chat_history or [],
            'context': None,
            'response': None,
            'retrieved_texts': None,
            'is_ice_breaker': False,
            'error': None
        }
        
        # Run the workflow graph
        self.workflow_result = self.graph.invoke(initial_state)
        
        # If there's an error, return error message
        if self.workflow_result.get('error'):
            yield f"Error: {self.workflow_result['error']}"
            return
        
        # If it's an ice breaker, yield the predefined response
        if self.workflow_result.get('is_ice_breaker'):
            yield self.workflow_result.get('response', 'Hello!')
            return
        
        # Generate and stream the response
        response_generator = self.response_generator.generate(
            retrieved_texts=self.workflow_result.get('retrieved_texts', []),
            query=input_text,
            previous_context=chat_history,
            llm=self.llm
        )
        
        # Yield tokens
        yield from response_generator
