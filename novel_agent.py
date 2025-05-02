#%%
from typing import TypedDict, List, Dict, Any, Literal, Optional, Annotated
import os
import operator
import logging
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from IPython.display import Image, display

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the models for structured output
class NovelRequirements(BaseModel):
    genre: str
    themes: List[str]
    total_chapters: int

class Character(BaseModel):
    name: str
    role: str
    background: str
    motivation: str
    arc: str

class CharacterList(BaseModel):
    characters: List[Character]

class Setting(BaseModel):
    name: str
    description: str
    significance: str

class SettingList(BaseModel):
    settings: List[Setting]

class ChapterOutline(BaseModel):
    chapter: int
    title: str
    summary: str
    key_events: List[str]
    character_development: Dict[str, str] = Field(None, description="A dictionary of character names as keys and their development as values")

class ChapterOutlineList(BaseModel):
    chapters: List[ChapterOutline]

class ChapterFeedback(BaseModel):
    pacing_issues: Optional[List[str]] = Field(default_factory=list)
    character_consistency: Optional[List[str]] = Field(default_factory=list)
    plot_holes: Optional[List[str]] = Field(default_factory=list)
    style_suggestions: Optional[List[str]] = Field(default_factory=list)
    overall_assessment: str

# Define the state for our novel writing process
class NovelState(TypedDict):
    user_input: str
    genre: str
    characters: Annotated[list, operator.add]
    plot_outline: Annotated[list, operator.add]
    current_chapter: int
    total_chapters: int
    settings: Annotated[list, operator.add]
    themes: List[str]
    feedback: Annotated[list, operator.add]
    current_draft: str
    final_output: str
    novel_output_file: str  # New field to store the output file path

# Initialize Azure OpenAI client if needed
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    raise ValueError("AZURE_OPENAI_API_KEY is not set")

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

NOVEL_LANGUAGE = "tamil"

# Prompts for novel writing process
requirements_prompt = """
You are a professional novelist and creative writing expert. Based on the user's input,
extract the following information:
1. The genre of the novel they want to write
2. Key themes they want to explore
3. How many chapters would be appropriate for this story (provide only a specific number, e.g., 10, 12, 15)

User input: {user_input}

Return this information in a structured format.
"""

character_development_prompt = """
You are a character development expert. Create compelling characters for a {genre} novel
that explores themes like {themes}.

For each character, provide:
- Name
- Role in the story (protagonist, antagonist, supporting)
- Background
- Motivation
- Character arc

Create at least 3 characters, including a protagonist and antagonist.

All character names and details should be appropriate for a novel written in {language} without any translations.
"""

settings_prompt = """
You are a world-building expert. Create detailed settings for a {genre} novel that explores
themes like {themes}.

For each setting, provide:
- Name
- Detailed description
- Significance to the story

Create at least 2 key settings for the novel.

All setting names and descriptions should be appropriate for a novel written in {language}.
"""

plot_outline_prompt = """
You are a plot development expert. Create a detailed plot outline for a {genre} novel with {total_chapters} chapters.
The novel explores themes like {themes} and features the following characters:

{characters}

The novel takes place in these settings:
{settings}

For each chapter, provide:
- Chapter number
- Title
- Summary
- Key events
- Character development (Should be a dictionary of character names as keys and their development as values)

Ensure the plot follows a coherent structure with rising action, climax, and resolution.
All chapter titles should be in {language}.
"""

chapter_writing_prompt = """
You are writing Chapter {chapter_num} of a {genre} novel. Here's the outline for this chapter:

{chapter_outline}

This chapter features the following characters:
{characters_in_chapter}

And takes place in these settings:
{settings_in_chapter}

Write a complete, engaging chapter that follows the outline and develops the characters according to their arcs.
Focus on showing rather than telling, using vivid descriptions, and balancing dialogue with action.

IMPORTANT: Write the entire chapter in {language}. Use appropriate idiomatic expressions, cultural references, 
and writing style that would be natural and engaging for native {language} readers.
"""

chapter_review_prompt = """
You are a professional editor reviewing Chapter {chapter_num} of a {genre} novel written in {language}.

The chapter outline:
{chapter_outline}

The written chapter:
{chapter_content}

Provide detailed feedback on:
- Pacing issues
- Character consistency
- Plot holes or inconsistencies
- Style and language suggestions
- Overall assessment

Be constructive and specific in your criticism, while being aware of the literary standards 
and stylistic conventions of novels written in {language}.
"""

chapter_revision_prompt = """
You are revising Chapter {chapter_num} of a {genre} novel based on editorial feedback.

Original chapter:
{chapter_content}

{feedback}

Revise the chapter to address all the feedback points while maintaining the overall narrative flow and tone.
Focus on strengthening character development, fixing inconsistencies, and improving the prose.

IMPORTANT: The chapter must be written entirely in {language}. Make sure your revision maintains 
the authentic style, flow, and cultural nuances of {language} literature.
"""

final_review_prompt = """
You are a professional editor reviewing a complete {genre} novel manuscript written in {language}.

The novel explores themes like {themes} and features the following characters:
{characters}

Provide a comprehensive review focusing on:
- Overall narrative arc
- Character development throughout the story
- Thematic cohesion
- Pacing and structure
- Style and voice consistency

Suggest any final edits or improvements that would make the novel ready for publication 
for a {language}-speaking audience.
"""

# Define the nodes for our novel writing process with actual LLM logic
def understand_requirements(state: NovelState) -> NovelState:
    """Analyze user input to understand the novel requirements"""
    logger.info("Starting understand_requirements node")
    logger.debug(f"Input state: {state}")
    
    prompt = requirements_prompt.format(user_input=state["user_input"])
    logger.debug(f"Generated prompt: {prompt}")
    
    response = llm.with_structured_output(NovelRequirements).invoke([HumanMessage(content=prompt)])
    logger.debug(f"LLM response: {response}")
    
    
    genre = response.genre
    themes = response.themes
    total_chapters = response.total_chapters
    
    # Create novel output file
    novel_output_file = "novel_in_progress.txt"
    with open(novel_output_file, 'w') as f:
        f.write(f"Novel Working Draft\nGenre: {genre}\nThemes: {', '.join(themes)}\nTotal Chapters: {total_chapters}\n\n")
    
    result = {
        **state,
        "genre": genre,
        "themes": themes,
        "total_chapters": total_chapters,
        "novel_output_file": novel_output_file
    }
    
    logger.info(f"Completed understand_requirements node: genre={genre}, themes={themes}, chapters={total_chapters}")
    logger.debug(f"Output state: {result}")
    
    return result

def develop_characters(state: NovelState) -> NovelState:
    """Create detailed character profiles for the novel"""
    logger.info("Starting develop_characters node")
    logger.debug(f"Input state: {state}")
    
    prompt = character_development_prompt.format(
        genre=state["genre"],
        themes=", ".join(state["themes"]),
        language=NOVEL_LANGUAGE
    )
    logger.debug(f"Generated prompt: {prompt}")
    
    response = llm.with_structured_output(CharacterList).invoke([HumanMessage(content=prompt)])
    logger.debug(f"LLM response: {response}")
    
    characters = response.characters
    
    result = {
        **state,
        "characters": characters
    }
    
    logger.info("Completed develop_characters node")
    logger.debug(f"Output state: {result}")
    
    return result

def create_settings(state: NovelState) -> NovelState:
    """Develop the world and settings for the novel"""
    logger.info("Starting create_settings node")
    logger.debug(f"Input state: {state}")
    
    prompt = settings_prompt.format(
        genre=state["genre"],
        themes=", ".join(state["themes"]),
        language=NOVEL_LANGUAGE
    )
    logger.debug(f"Generated prompt: {prompt}")
    
    response = llm.with_structured_output(SettingList).invoke([HumanMessage(content=prompt)])
    logger.debug(f"LLM response: {response}")
    
    settings = response.settings
    
    result = {
        **state,
        "settings": settings
    }
    
    logger.info("Completed create_settings node")
    logger.debug(f"Output state: {result}")
    
    return result

            

def outline_plot(state: NovelState) -> NovelState:
    """Create a detailed plot outline with story arcs"""
    logger.info("Starting outline_plot node")
    logger.debug(f"Input state: {state}")
    
    characters_text = "\n".join([
        f"- {char.name}: {char.role}, {char.motivation}" for char in state["characters"]
    ])
    
    settings_text = "\n".join([
        f"- {setting.name}: {setting.description[:100]}..." for setting in state["settings"]
    ])
    
    prompt = plot_outline_prompt.format(
        genre=state["genre"],
        total_chapters=state["total_chapters"],
        themes=", ".join(state["themes"]),
        characters=characters_text,
        settings=settings_text,
        language=NOVEL_LANGUAGE
    )
    logger.debug(f"Generated prompt: {prompt}")
    
    response = llm.with_structured_output(ChapterOutlineList).invoke([HumanMessage(content=prompt)])
    plot_outline = response.chapters
    
    result = {
        **state,
        "plot_outline": plot_outline,
        "current_chapter": 1
    }
    
    logger.info("Completed outline_plot node")
    logger.debug(f"Output state: {result}")
    
    return result

def write_chapter(state: NovelState) -> NovelState:
    """Write the current chapter of the novel"""
    logger.info(f"Starting write_chapter node for chapter {state['current_chapter']}")
    logger.debug(f"Input state: {state}")
    
    current_chapter = state["current_chapter"]
    chapter_outline = next((ch for ch in state["plot_outline"] if ch.chapter == current_chapter), None)
    
    if not chapter_outline:
        logger.error(f"No outline found for chapter {current_chapter}")
        return state
    
    # Find characters involved in this chapter
    characters_in_chapter = []
    for char_name in chapter_outline.character_development.keys():
        char = next((c for c in state["characters"] if c.name == char_name), None)
        if char:
            characters_in_chapter.append(char)
    
    characters_text = "\n".join([
        f"- {char.name}: {char.role}" for char in characters_in_chapter
    ])
    
    # Determine settings for this chapter (simplified)
    settings_in_chapter = state["settings"][:2]  # Just use the first two settings
    settings_text = "\n".join([
        f"- {setting.name}: {setting.description[:100]}..." for setting in settings_in_chapter
    ])
    
    # Format chapter outline as text
    chapter_outline_text = f"""Chapter {chapter_outline.chapter}: {chapter_outline.title}
Summary: {chapter_outline.summary}

Key Events:
"""
    for event in chapter_outline.key_events:
        chapter_outline_text += f"- {event}\n"
        
    chapter_outline_text += "\nCharacter Development:\n"
    for char_name, development in chapter_outline.character_development.items():
        chapter_outline_text += f"- {char_name}: {development}\n"
    
    prompt = chapter_writing_prompt.format(
        chapter_num=current_chapter,
        genre=state["genre"],
        chapter_outline=chapter_outline_text,
        characters_in_chapter=characters_text,
        settings_in_chapter=settings_text,
        language=NOVEL_LANGUAGE
    )
    logger.debug(f"Generated prompt: {prompt}")
    
    logger.info(f"Generating content for chapter {current_chapter}: {chapter_outline.title}")
    chapter_content = llm.invoke([HumanMessage(content=prompt)]).content
    logger.debug(f"Generated chapter content length: {len(chapter_content)} characters")
    
    # Append to current draft
    current_draft = state.get("current_draft", "")
    chapter_formatted = f"\n\nCHAPTER {current_chapter}: {chapter_outline.title}\n\n{chapter_content}"
    updated_draft = current_draft + chapter_formatted
    
    # Import datetime for timestamp
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Append the chapter to the novel output file
    novel_output_file = state.get("novel_output_file", "novel_in_progress.txt")
    with open(novel_output_file, 'a') as f:
        f.write(f"\n--- CHAPTER {current_chapter}: {chapter_outline.title} ---\n\n")
        f.write(f"Draft created on: {current_time}\n\n")
        f.write(chapter_content)
        f.write("\n\n--------------------\n\n")
    
    logger.info(f"Appended chapter {current_chapter} to {novel_output_file}")
    
    result = {
        **state,
        "current_draft": updated_draft
    }
    
    logger.info(f"Completed write_chapter node for chapter {current_chapter}")
    logger.debug(f"Output state length: {len(result['current_draft'])} characters")
    
    return result

def review_chapter(state: NovelState) -> NovelState:
    """Review and edit the current chapter"""
    logger.info(f"Starting review_chapter node for chapter {state['current_chapter']}")
    logger.debug(f"Input state: {state}")
    
    current_chapter = state["current_chapter"]
    chapter_outline = next((ch for ch in state["plot_outline"] if ch.chapter == current_chapter), None)
    
    if not chapter_outline:
        logger.error(f"No outline found for chapter {current_chapter}")
        return state
    
    # Extract just the current chapter from the draft
    draft_parts = state["current_draft"].split(f"\n\nCHAPTER {current_chapter}:")
    if len(draft_parts) < 2:
        logger.error(f"Could not find chapter {current_chapter} in the draft")
        return state
    
    current_chapter_content = draft_parts[-1]
    if "\n\nCHAPTER " in current_chapter_content:
        current_chapter_content = current_chapter_content.split("\n\nCHAPTER ")[0]
    
    logger.debug(f"Extracted chapter content length: {len(current_chapter_content)} characters")
    
    # Format chapter outline as text
    chapter_outline_text = f"""Chapter {chapter_outline.chapter}: {chapter_outline.title}
Summary: {chapter_outline.summary}

Key Events:
"""
    for event in chapter_outline.key_events:
        chapter_outline_text += f"- {event}\n"
        
    chapter_outline_text += "\nCharacter Development:\n"
    for char_name, development in chapter_outline.character_development.items():
        chapter_outline_text += f"- {char_name}: {development}\n"
    
    prompt = chapter_review_prompt.format(
        chapter_num=current_chapter,
        genre=state["genre"],
        chapter_outline=chapter_outline_text,
        chapter_content=current_chapter_content,
        language=NOVEL_LANGUAGE
    )
    logger.debug(f"Generated prompt: {prompt}")
    
    logger.info(f"Reviewing chapter {current_chapter}: {chapter_outline.title}")
    response = llm.with_structured_output(ChapterFeedback).invoke([HumanMessage(content=prompt)])
    logger.debug(f"LLM response: {response}")
    
    
    result = {
        **state,
        "feedback": [response]
    }
    
    logger.info(f"Completed review_chapter node for chapter {current_chapter}")
    logger.debug(f"Output state updated with feedback")
    
    return result

def revise_chapter(state: NovelState) -> NovelState:
    """Revise the chapter based on feedback"""
    logger.info(f"Starting revise_chapter node for chapter {state['current_chapter']}")
    logger.debug(f"Input state: {state}")
    
    current_chapter = state["current_chapter"]
    
    # Extract just the current chapter from the draft
    draft_parts = state["current_draft"].split(f"\n\nCHAPTER {current_chapter}:")
    if len(draft_parts) < 2:
        logger.error(f"Could not find chapter {current_chapter} in the draft")
        return state
    
    chapter_start = f"\n\nCHAPTER {current_chapter}:"
    chapter_content_with_title = draft_parts[-1]
    
    if "\n\nCHAPTER " in chapter_content_with_title:
        chapter_content_with_title = chapter_content_with_title.split("\n\nCHAPTER ")[0]
    
    logger.debug(f"Extracted chapter content length: {len(chapter_content_with_title)} characters")
    
    feedback = state["feedback"][-1]
    logger.debug(f"Using feedback: {feedback}")
    
    # Convert feedback to text format instead of using json()
    feedback_text = "Editorial Feedback:\n"
    
    if feedback.pacing_issues:
        feedback_text += "\nPacing Issues:\n"
        for issue in feedback.pacing_issues:
            feedback_text += f"- {issue}\n"
            
    if feedback.character_consistency:
        feedback_text += "\nCharacter Consistency:\n"
        for issue in feedback.character_consistency:
            feedback_text += f"- {issue}\n"
            
    if feedback.plot_holes:
        feedback_text += "\nPlot Holes/Inconsistencies:\n"
        for issue in feedback.plot_holes:
            feedback_text += f"- {issue}\n"
            
    if feedback.style_suggestions:
        feedback_text += "\nStyle Suggestions:\n"
        for suggestion in feedback.style_suggestions:
            feedback_text += f"- {suggestion}\n"
            
    feedback_text += f"\nOverall Assessment:\n{feedback.overall_assessment}\n"
    
    prompt = chapter_revision_prompt.format(
        chapter_num=current_chapter,
        genre=state["genre"],
        chapter_content=chapter_content_with_title,
        feedback=feedback_text,
        language=NOVEL_LANGUAGE
    )
    logger.debug(f"Generated prompt: {prompt}")
    
    logger.info(f"Revising chapter {current_chapter} based on feedback")
    revised_chapter = llm.invoke([HumanMessage(content=prompt)]).content
    logger.debug(f"Revised chapter content length: {len(revised_chapter)} characters")
    
    # Replace the chapter in the draft
    updated_draft = state["current_draft"].replace(
        chapter_start + chapter_content_with_title,
        chapter_start + revised_chapter
    )
    
    # Get the chapter title from plot outline
    chapter_outline = next((ch for ch in state["plot_outline"] if ch.chapter == current_chapter), None)
    chapter_title = chapter_outline.title if chapter_outline else f"Chapter {current_chapter}"
    
    # Import datetime for timestamp
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Append the revised chapter to the novel output file
    novel_output_file = state.get("novel_output_file", "novel_in_progress.txt")
    with open(novel_output_file, 'a') as f:
        f.write(f"\n--- REVISED CHAPTER {current_chapter}: {chapter_title} ---\n\n")
        f.write(f"Revision created on: {current_time}\n\n")
        f.write(revised_chapter)
        f.write("\n\n--------------------\n\n")
    
    logger.info(f"Appended revised chapter {current_chapter} to {novel_output_file}")
    
    result = {
        **state,
        "current_draft": updated_draft
    }
    
    logger.info(f"Completed revise_chapter node for chapter {current_chapter}")
    logger.debug(f"Output state with revised draft length: {len(result['current_draft'])} characters")
    
    return result

def check_if_more_chapters(state: NovelState) -> Literal["next_chapter", "finalize"]:
    """Check if there are more chapters to write"""
    logger.info(f"Checking if more chapters after chapter {state['current_chapter']} of {state['total_chapters']}")
    
    if state["current_chapter"] < state["total_chapters"]:
        logger.info(f"More chapters remaining, moving to next chapter")
        return "next_chapter"
    else:
        logger.info(f"All chapters complete, proceeding to finalize")
        return "finalize"

def increment_chapter(state: NovelState) -> NovelState:
    """Move to the next chapter"""
    logger.info(f"Incrementing chapter from {state['current_chapter']} to {state['current_chapter'] + 1}")
    
    result = {
        **state,
        "current_chapter": state["current_chapter"] + 1
    }
    
    logger.debug(f"Updated state with incremented chapter number")
    return result

def finalize_novel(state: NovelState) -> NovelState:
    """Finalize the novel with conclusion and polish"""
    logger.info("Starting finalize_novel node")
    logger.debug(f"Input state: {state}")
    
    characters_text = "\n".join([
        f"- {char.name}: {char.role}, {char.arc}" for char in state["characters"]
    ])
    
    prompt = final_review_prompt.format(
        genre=state["genre"],
        themes=", ".join(state["themes"]),
        characters=characters_text,
        language=NOVEL_LANGUAGE
    )
    logger.debug(f"Generated prompt: {prompt}")
    
    logger.info("Generating final review for the complete novel")
    final_review = llm.invoke([HumanMessage(content=prompt)]).content
    logger.debug(f"Final review length: {len(final_review)} characters")
    
    final_output = state["current_draft"] + "\n\n--- EDITOR'S FINAL NOTES ---\n\n" + final_review
    
    # Append the final review to the novel output file
    novel_output_file = state.get("novel_output_file", "novel_in_progress.txt")
    with open(novel_output_file, 'a') as f:
        f.write("\n\n==============================================\n")
        f.write("============== EDITOR'S FINAL NOTES ==============\n")
        f.write("==============================================\n\n")
        f.write(final_review)
        f.write("\n\n--- END OF NOVEL ---\n")
    
    logger.info(f"Appended final review to {novel_output_file}")
    
    # Create a final compiled version with .finished suffix
    final_novel_file = novel_output_file.replace(".txt", ".finished.txt")
    with open(final_novel_file, 'w') as f:
        f.write(final_output)
    
    logger.info(f"Created final compiled novel at {final_novel_file}")
    
    result = {
        **state,
        "final_output": final_output
    }
    
    logger.info("Completed finalize_novel node")
    logger.debug(f"Final output total length: {len(result['final_output'])} characters")
    
    return result

# Build the graph
builder = StateGraph(NovelState)

# Add nodes
builder.add_node("understand_requirements", understand_requirements)
builder.add_node("develop_characters", develop_characters)
builder.add_node("create_settings", create_settings)
builder.add_node("outline_plot", outline_plot)
builder.add_node("write_chapter", write_chapter)
builder.add_node("review_chapter", review_chapter)
builder.add_node("revise_chapter", revise_chapter)
builder.add_node("increment_chapter", increment_chapter)
builder.add_node("finalize_novel", finalize_novel)

# Add edges
builder.add_edge(START, "understand_requirements")
builder.add_edge("understand_requirements", "develop_characters")
builder.add_edge("develop_characters", "create_settings")
builder.add_edge("create_settings", "outline_plot")
builder.add_edge("outline_plot", "write_chapter")
builder.add_edge("write_chapter", "review_chapter")
builder.add_edge("review_chapter", "revise_chapter")
builder.add_conditional_edges(
    "revise_chapter", 
    check_if_more_chapters,
    {
        "next_chapter": "increment_chapter",
        "finalize": "finalize_novel"
    }
)
builder.add_edge("increment_chapter", "write_chapter")
builder.add_edge("finalize_novel", END)

# Compile the graph
novel_writing_graph = builder.compile()

#%%

# Visualize the graph
display(Image(novel_writing_graph.get_graph().draw_mermaid_png()))
# %%

novel_writing_graph.invoke({
    "user_input": "I want to write a tamil novel about a college student who is a software engineer and a hacker",
    "novel_output_file": "novel_in_progress.txt"
})
# %%
