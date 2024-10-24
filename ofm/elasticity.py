from dataclasses import dataclass
from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np
@dataclass
class ElasticRange:
    """Define range and constraints for elastic parameters."""
    min_val: Union[int, float]
    max_val: Union[int, float]
    step: Union[int, float] = 1  # Step size for sampling
    constraints: Optional[List[str]] = None  # e.g., ["divisible_by_8"]
    
    def is_valid(self, value: Union[int, float]) -> bool:
        """Check if a value satisfies the range and constraints."""
        if not (self.min_val <= value <= self.max_val):
            return False
            
        if self.constraints:
            for constraint in self.constraints:
                if constraint == "divisible_by_8" and value % 8 != 0:
                    return False
                # Add more constraints as needed
                
        return True
    
    def sample(self) -> Union[int, float]:
        """Sample a valid value from the range."""
        if isinstance(self.min_val, int) and isinstance(self.max_val, int):
            possible_values = np.arange(self.min_val, self.max_val + 1, self.step)
            if self.constraints:
                possible_values = [v for v in possible_values if self.is_valid(v)]
            return int(np.random.choice(possible_values))
        else:
            value = np.random.uniform(self.min_val, self.max_val)
            return round(value / self.step) * self.step
