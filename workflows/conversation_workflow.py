from typing import Dict, Any, List, Optional, Union, Generator

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

# Import local modules for retrieval and generation
from retrieval.retrieval_system import RetrievalSystem
from generation.generate_response import ResponseGenerator
from agents.ice_breaker_agent import IceBreakerAgent
import logging

logger = logging.getLogger(__name__)

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
        return result['response']

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

            error: Optional[str]

        graph = StateGraph(ConversationState)

        # Define nodes for different stages of conversation processing
        graph.add_node("check_ice_breaker", self.check_ice_breaker)
        graph.add_node("retrieve_context", self.retrieve_context)
        graph.add_node("generate_response", self.generate_response)
        graph.add_node("handle_error", self.handle_error)

        # Set the entry point based on whether ice breaker is enabled
        if self.enable_ice_breaker:
            graph.set_entry_point("check_ice_breaker")
            
            # Conditional routing based on ice breaker check
            graph.add_conditional_edges(
                "check_ice_breaker",
                self.route_after_ice_breaker,
                {
                    "handle_error": "handle_error",
                    "retrieve_context": "retrieve_context",
                    END: END
                }
            )
        else:
            # If ice breaker is disabled, start directly with retrieve_context
            graph.set_entry_point("retrieve_context")

        # Connect nodes for normal workflow
        graph.add_edge("retrieve_context", "generate_response")
        
        graph.add_conditional_edges(
            "generate_response",
            self.route_response,
            {
                "handle_error": "handle_error",
                END: END
            }
        )

        graph.add_edge("handle_error", END)

        return graph.compile()

    def check_ice_breaker(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the input is an ice breaker. If ice breaker is enabled,
        let the LLM decide whether to use it.
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with ice breaker check result
        """
        # If ice breaker is disabled, skip the check entirely
        if not self.enable_ice_breaker:
            return {**state}

        try:
            input_text = state.get('input', '')
            
            # Let the LLM decide whether to use the ice breaker tool
            messages = [
                SystemMessage(content=(
                    "You are a helpful assistant. If the user's input appears to be a greeting "
                    "or casual conversation starter, use the ice_breaker_detector tool to handle it. "
                    "Otherwise, proceed with normal processing."
                )),
                HumanMessage(content=input_text)
            ]
            
            response = self.llm.invoke(messages)
            
            # If the LLM used the ice breaker tool and got a response
            if hasattr(response, 'additional_kwargs') and response.additional_kwargs.get('tool_calls'):
                tool_call = response.additional_kwargs['tool_calls'][0]
                if tool_call['function']['name'] == self.ice_breaker_tool.name:
                    ice_breaker_result = self.ice_breaker_agent.process(input_text)
                    return {
                        **state, 
                        'response': ice_breaker_result.get('response')
                    }

            return {**state}

        except Exception as e:
            logger.error(f"Error in ice breaker check: {e}")
            return {**state, 'error': str(e)}

    def route_after_ice_breaker(self, state: Dict[str, Any]) -> str:
        """
        Determine next step after ice breaker check
        
        Args:
            state: Current conversation state
        
        Returns:
            Next node to execute
        """
        if state.get('error'):
            return "handle_error"
        elif state.get('response'):
            return END
        else:
            return "retrieve_context"

    def retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant context for the input
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with retrieved context
        """
        try:
            input_text = state.get('input', '')
            retrieved_texts = self.retrieval_system.retrieve(input_text)
            return {
                **state,
                'retrieved_texts': retrieved_texts or [],
                'context': '\n\n'.join(retrieved_texts) if retrieved_texts else ''
            }
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
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
        return 'handle_error' if state.get('error') else END

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
            'retrieved_texts': [],
            'error': None
        }
        
        # Run the workflow graph
        self.workflow_result = self.graph.invoke(initial_state)
        
        # If there's an error, return error message
        if self.workflow_result.get('error'):
            yield f"Error: {self.workflow_result['error']}"
            return
        
        # If it's an ice breaker response, yield it
        if self.workflow_result.get('response'):
            yield self.workflow_result['response']
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
