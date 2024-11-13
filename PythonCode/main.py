from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import discord
import spacy

# Load environment variables
load_dotenv()

# Initialize the OpenAI client with API key and base URL
client = OpenAI(
    api_key=os.getenv("RUNPOD_TOKEN"),
    base_url=os.getenv("RUNPOD_CHATBOT_URL")
)
client_discord=discord.Client(intents=discord.Intents.default())

nlp=spacy.load("en_core_web_sm")

# Set model name from environment variable
model_name = os.getenv("MODEL_NAME")

def get_promt_for_algorithms(algoName):
    prompts={
        "binary search": (
            "You are a helpful assistant that explains the Binary Search algorithm. "
            "Break down each step in a structured JSON format, and provide a summary at the end. "
            "Output only a JSON object with each step clearly outlined, and no additional text.\n\n"
            "Example JSON format:\n"
            "{\n"
            "  \"algorithm\": \"Binary Search\",\n"
            "  \"steps\": [\n"
            "    {\"step\": 1, \"description\": \"Initialize pointers at the start and end of the array.\"},\n"
            "    {\"step\": 2, \"description\": \"Check if the middle element is the target.\"},\n"
            "    {\"step\": 3, \"description\": \"If target is smaller, adjust end pointer.\"},\n"
            "    {\"step\": 4, \"description\": \"If target is larger, adjust start pointer.\"},\n"
            "    {\"step\": 5, \"description\": \"Repeat until target is found or pointers cross.\"}\n"
            "  ],\n"
            "  \"summary\": \"Binary search finds a target in a sorted array by dividing the interval in half.\""
            "\n}"
        ),
        "merge_sort": (
    "You are a helpful assistant that explains the Merge Sort algorithm. "
    "Break down each step in a structured JSON format, and provide a summary at the end. "
    "Output only a JSON object with each step clearly outlined, and no additional text.\n\n"
    "Example JSON format:\n"
    "{\n"
    "  \"algorithm\": \"Merge Sort\",\n"
    "  \"steps\": [\n"
    "    {\"step\": 1, \"description\": \"If array length is less than 2, return the array.\"},\n"
    "    {\"step\": 2, \"description\": \"Split the array into two halves.\"},\n"
    "    {\"step\": 3, \"description\": \"Recursively apply merge sort on each half.\"},\n"
    "    {\"step\": 4, \"description\": \"Merge the sorted halves back together.\"},\n"
    "    {\"step\": 5, \"description\": \"Continue merging until a single sorted array remains.\"}\n"
    "  ],\n"
    "  \"summary\": \"Merge sort divides the array and recursively sorts each half before merging.\""
    "\n}"
),
        "quick_sort": (
            "You are a helpful assistant that explains the Quick Sort algorithm. "
            "Break down each step in a structured JSON format, and provide a summary at the end. "
            "Output only a JSON object with each step clearly outlined, and no additional text.\n\n"
            "Example JSON format:\n"
            "{\n"
            "  \"algorithm\": \"Quick Sort\",\n"
            "  \"steps\": [\n"
            "    {\"step\": 1, \"description\": \"Choose a pivot element from the array.\"},\n"
            "    {\"step\": 2, \"description\": \"Partition elements around the pivot.\"},\n"
            "    {\"step\": 3, \"description\": \"Place elements smaller than pivot to its left and larger to its right.\"},\n"
            "    {\"step\": 4, \"description\": \"Recursively apply quick sort to left and right partitions.\"},\n"
            "    {\"step\": 5, \"description\": \"Repeat until the entire array is sorted.\"}\n"
            "  ],\n"
            "  \"summary\": \"Quick sort sorts an array by partitioning around a pivot element and recursively sorting.\""
            "\n}"
        ),
        "dfs": (
            "You are a helpful assistant that explains the Depth-First Search (DFS) algorithm. "
            "Break down each step in a structured JSON format, and provide a summary at the end. "
            "Output only a JSON object with each step clearly outlined, and no additional text.\n\n"
            "Example JSON format:\n"
            "{\n"
            "  \"algorithm\": \"Depth-First Search\",\n"
            "  \"steps\": [\n"
            "    {\"step\": 1, \"description\": \"Start from the root node or initial node.\"},\n"
            "    {\"step\": 2, \"description\": \"Mark the node as visited.\"},\n"
            "    {\"step\": 3, \"description\": \"Explore each unvisited neighbor recursively.\"},\n"
            "    {\"step\": 4, \"description\": \"Backtrack when reaching a node with no unvisited neighbors.\"},\n"
            "    {\"step\": 5, \"description\": \"Repeat until all nodes have been visited.\"}\n"
            "  ],\n"
            "  \"summary\": \"DFS explores as far as possible down each branch before backtracking.\""
            "\n}"
        ),
        "bfs": (
            "You are a helpful assistant that explains the Breadth-First Search (BFS) algorithm. "
            "Break down each step in a structured JSON format, and provide a summary at the end. "
            "Output only a JSON object with each step clearly outlined, and no additional text.\n\n"
            "Example JSON format:\n"
            "{\n"
            "  \"algorithm\": \"Breadth-First Search\",\n"
            "  \"steps\": [\n"
            "    {\"step\": 1, \"description\": \"Start from the root node or initial node.\"},\n"
            "    {\"step\": 2, \"description\": \"Mark the node as visited and add it to the queue.\"},\n"
            "    {\"step\": 3, \"description\": \"Dequeue a node and explore each unvisited neighbor.\"},\n"
            "    {\"step\": 4, \"description\": \"Add each unvisited neighbor to the queue.\"},\n"
            "    {\"step\": 5, \"description\": \"Repeat until the queue is empty and all nodes have been visited.\"}\n"
            "  ],\n"
            "  \"summary\": \"BFS explores each level of a graph or tree before moving to the next level.\""
            "\n}"
        ),
    }
    return prompts.get(algoName.lower())

def identify_algo(query):
    doc = nlp(query.lower())
    algorithm_keywords = {
        "binary search": ["binary search", "find in sorted array", "search sorted array"],
        "quick sort": ["quick sort", "quicksort", "sort array quickly", "fast sorting"],
        "merge sort": ["merge sort", "merge array", "divide and conquer sort"],
        "depth-first search": ["depth-first search", "dfs", "graph dfs", "tree dfs"],
        "breadth-first search": ["breadth-first search", "bfs", "graph bfs", "tree bfs"],
        "dynamic programming": ["dynamic programming", "dp", "optimize subproblems", "dynamic solution"],
        "dijkstra's algorithm": ["dijkstra", "shortest path", "graph shortest path", "dijkstra's algorithm"],
        "a* search": ["a* search", "a-star", "astar", "pathfinding"],
        "floyd-warshall": ["floyd-warshall", "all pairs shortest path", "graph pathfinding"],
        "bellman-ford": ["bellman-ford", "single source shortest path", "graph shortest path"],
        "prim's algorithm": ["prim's algorithm", "minimum spanning tree", "mst prim"],
        "kruskal's algorithm": ["kruskal's algorithm", "minimum spanning tree", "mst kruskal"],
        "topological sort": ["topological sort", "dag sort", "dependency sort"],
        "knapsack problem": ["knapsack", "dp knapsack", "optimize knapsack"],
        "fibonacci sequence (dp)": ["fibonacci", "fibonacci sequence", "dp fibonacci"],
        "longest common subsequence": ["longest common subsequence", "lcs", "dp lcs"],
        "longest increasing subsequence": ["longest increasing subsequence", "lis", "dp lis"],
        "counting sort": ["counting sort", "non-comparative sort", "integer sort"],
        "radix sort": ["radix sort", "integer sort", "digit-based sort"],
        "heap sort": ["heap sort", "heapify", "priority queue sort"],
        "insertion sort": ["insertion sort", "sort by insertion", "stable sort"],
        "selection sort": ["selection sort", "sort by selection", "array selection"],
        "bubble sort": ["bubble sort", "simple sort", "exchange sort"],
        "bellman-ford": ["bellman-ford", "single source shortest path", "graph shortest path"],
        "backtracking": ["backtracking", "recursive search", "constraint satisfaction"],
        "divide and conquer": ["divide and conquer", "recursive division", "problem partition"],
        "greedy algorithm": ["greedy algorithm", "optimal local choice", "greedy approach"],
        "branch and bound": ["branch and bound", "bnb", "recursive optimization"],
        "sliding window": ["sliding window", "window technique", "contiguous subarray"],
    }
    for token in doc:
        for algo_name,keywords in algorithm_keywords.items():
            if any(keyword in token.text for keyword in keywords):
                return algo_name
        return None


def get_algorithm_explanation(algorithm_name):
    prompt = get_promt_for_algorithms(algorithm_name)
    if not prompt:
        return {"error": f"No explanation template found for {algorithm_name}"}

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Explain the {algorithm_name} algorithm step-by-step in JSON format."}
    ]

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=2000
    )
    response_text = response.choices[0].message.content


    try:
        response_json = json.loads(response_text)
        return response_json
    except json.JSONDecodeError:
        return {"error": "Response was not in JSON format", "response": response_text}


@client_discord.event
async def on_ready():
    print(f"{client_discord.user} has connected to Discord!")


@client_discord.event
async def on_message(message):
    if message.author == client_discord.user:
        return

    if message.content.startswith("!explain"):

        algorithm_name = message.content[len("!explain "):].strip()

        if not algorithm_name:
            await message.channel.send("Please provide an algorithm name. Example: `!explain binary search`")
            return


        explanation = get_algorithm_explanation(algorithm_name)

        if "error" in explanation:
            await message.channel.send(explanation["error"])
        else:

            explanation_text = json.dumps(explanation, indent=2)
            await message.channel.send(f"```json\n{explanation_text}\n```")


# Run the Discord bot
client_discord.run(os.getenv("DISCORD_TOKEN"))