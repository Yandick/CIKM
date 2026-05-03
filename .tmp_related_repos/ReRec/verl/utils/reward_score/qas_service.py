from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import networkx as nx
from typing import Optional
from verl.utils.reward_score.rec import extract_recommendation
import time
import argparse
from contextlib import asynccontextmanager

# Global variable to store graph
_G = None
_dataset_type = "book"  # Default dataset type

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Service lifecycle management"""
    # Load graph on startup
    global _G
    try:
        start_time = time.time()
        # Select different graph files based on dataset type
        if _dataset_type == "movie":
            graph_file = 'data/movie_item_attribute_graph.gexf'
        elif _dataset_type == "book":
            graph_file = 'data/book_item_attribute_graph.gexf'
        else:
            raise ValueError(f"Invalid dataset type: {_dataset_type}")
        
        _G = nx.read_gexf(graph_file)
        print(f"Graph loaded successfully from {graph_file}, take {time.time() - start_time} seconds")
    except Exception as e:
        raise Exception(f"Failed to load graph: {str(e)}")
    yield
    # Clean up resources on shutdown
    _G = None

# Create FastAPI application instance
app = FastAPI(lifespan=lifespan)

def compute_qas(G, recommended_item, target_items, epsilon=0.01):
    """
    Compute Query Alignment Score (QAS) for a recommended item against target items.
    
    Args:
        G (nx.Graph): Bipartite item-attribute graph.
        recommended_item (str): The recommended item.
        target_items (set/list): Set of target items.
        epsilon (float): Small constant for inverse degree centrality.
        
    Returns:
        float: QAS value.
    """
    def get_attribute_neighborhood(G, items, mode='recommendation'):
        """
        Get attribute sets connected to items
        
        Args:
            G (nx.Graph): Bipartite item-attribute graph.
            items: Single item (str) or set of items (set/list).
            mode (str): 'recommendation' or 'target', used to distinguish different processing modes
                - 'recommendation': Return all attributes of recommended item
                - 'target': Return shared attributes of target items
            
        Returns:
            set: Return corresponding attribute sets based on mode
        """
        if isinstance(items, str):
            items = {items}
        
        if mode == 'recommendation':
            # Recommendation mode: return all attributes of the recommended item
            attributes = set()
            for item in items:
                attributes.update(G.neighbors(item))
            return attributes
        else:  # mode == 'target'
            # Target mode: return shared attributes across all target items
            shared_attributes = None
            for item in items:
                if item not in G.nodes:
                    # print(f"item not in G.nodes: {item}")
                    continue
                item_attributes = set(G.neighbors(item))
                if shared_attributes is None:
                    shared_attributes = item_attributes
                else:
                    shared_attributes = shared_attributes.intersection(item_attributes)
            return shared_attributes if shared_attributes is not None else set()
        
    
    if recommended_item not in G.nodes:
        print(f"recommended_item not in G.nodes: {recommended_item}")
        return -1
    # Get attribute neighborhoods
    A_rec = get_attribute_neighborhood(G, recommended_item)
    # print(f"A_rec: {A_rec}")
    if type(target_items) == str:
        target_items = [target_items]
    A_target = get_attribute_neighborhood(G, target_items, mode='target')
    # print(f"A_target: {A_target}")
    # Get shared attributes
    shared_attributes = A_rec.intersection(A_target)
    # print(f"shared_attributes: {shared_attributes}")
    # Compute weights for all attributes in A_target
    # weights = compute_attribute_weights(G, A_target, epsilon)
    
    # # Compute numerator and denominator
    # numerator = sum(weights[attr] for attr in shared_attributes)
    # denominator = sum(weights[attr] for attr in A_target)

    numerator = len(shared_attributes)
    denominator = len(A_target)
    
    # Avoid division by zero
    if denominator == 0:
        return 0.0
    return numerator / denominator

def compute_qas_score(G, solution_str: str, ground_truth: str) -> float:
    """
    Compute the Query Alignment Score (QAS) between a recommended answer and ground truth.

    Args:
        G: Item-attribute bipartite graph.
        solution_str (str): Model output string containing the recommended item.
        ground_truth (str): Semicolon-separated ground truth item string.

    Returns:
        float: QAS score.
    """

    # Extract recommended item and target item
    recommendation = extract_recommendation(solution_str)

    if recommendation is None:
        # print(f"recommendation is None")
        return 0
        
    recommended_item = recommendation.strip()
    target_item = ground_truth.strip()
    # print(f"recommended_item: {recommended_item}, target_item: {target_item}")
    
    # 计算每个推荐电影的QAS并取平均值

    qas = compute_qas(G, recommended_item, target_item)

    # If no valid QAS score, return 0
    if not qas or qas == -1:
        return 0
        
    return qas


class QASRequest(BaseModel):
    solution_str: str
    ground_truth: str

class QASResponse(BaseModel):
    qas_score: float

@app.post("/compute_qas", response_model=QASResponse)
async def compute_qas_api(request: QASRequest):
    """API endpoint for computing QAS score."""
    if _G is None:
        raise HTTPException(status_code=500, detail="Graph not loaded")
    
    try:
        qas_score = compute_qas_score(_G, request.solution_str, request.ground_truth)
        return QASResponse(qas_score=qas_score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="QAS Service")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["movie", "book"], 
        default="movie",
        help="Dataset type to load (movie or book), default: movie"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the service, default: 0.0.0.0"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the service, default: 8000"
    )
    
    args = parser.parse_args()
    
    # Set global dataset type
    _dataset_type = args.dataset
    
    print(f"Starting QAS service with dataset: {_dataset_type}")
    uvicorn.run("verl.utils.reward_score.qas_service:app", host=args.host, port=args.port, reload=False)