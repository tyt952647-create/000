import numpy as np
import time
from collections import deque

class PrioritySelector:
    """
    BALANCED STRATEGIC FIRING SYSTEM
    
    Philosophy: Fire with CONFIDENCE on proven patterns, not random spray.
    
    - Only fire targets with CLEAR strategic value
    - Use past confidence (65%+) to guide decisions
    - Avoid solo small targets unless cluster opportunity nearby
    - Every 30s: fire on high-confidence targets only, never marginal ones
    - Learn and improve: track what works, repeat it
    """
    
    def __init__(self, fish_memory, shot_tracker):
        self.fish_memory = fish_memory
        self.shot_tracker = shot_tracker
        
        self.last_shot_time = time.time()
        self.mandatory_fire_interval = 30
        self.fire_aggression = 0
        
        # Confidence thresholds - STRICT
        self.min_killshot_confidence = 0.65
        self.min_convergence_confidence = 0.65
        self.min_cluster_size = 3  # Only burst clusters of 3+
        
        # Value thresholds - NO MARGINAL SHOTS
        self.min_efficiency = 1.0  # Must be break-even or better
        
        # Stats
        self.total_value_gained = 0
        self.total_shots_fired = 0
        self.efficiency_ratio = 1.0
        self.wasted_shots = 0
    
    def rank_targets(self, fishes, region, current_score):
        """
        Return targets ranked by CONFIDENCE + VALUE, not desperation.
        
        Only suggest firing if:
        1. Target has proven pattern (65%+ from history)
        2. OR convergence is geometrically likely (65%+ confidence)
        3. OR cluster of 3+ with positive efficiency
        4. AND expected gain >= 0
        """
        ranked = []
        
        time_since_last_shot = time.time() - self.last_shot_time
        must_fire_soon = time_since_last_shot > self.mandatory_fire_interval
        
        # Escalate confidence only mildly
        if time_since_last_shot > self.mandatory_fire_interval:
            self.fire_aggression = 1  # Slightly more aggressive at 30s+
        else:
            self.fire_aggression = 0  # Normal
        
        # ===== TIER 1: HIGH-CONFIDENCE KILLSHOTS =====
        # 65%+ success rate = FIRE (not all shots, only proven classes)
        p1_targets = self._rank_high_confidence_killshots(fishes, current_score)
        ranked.extend(p1_targets)
        
        # ===== TIER 2: STRONG CONVERGENCE =====
        # 2+ targets + 65%+ geometric confidence + positive value
        p2_targets = self._rank_convergence_confidence(fishes, region, current_score)
        ranked.extend(p2_targets)
        
        # ===== TIER 3: EFFICIENT CLUSTERS =====
        # 3+ small targets + efficiency > 1.0 = shared ammo wins
        p3_targets = self._rank_cluster_efficiency(fishes, current_score)
        ranked.extend(p3_targets)
        
        # ===== TIER 4: HIGH-VALUE SINGLES =====
        # Large fish with value > 2x cost (proven safe shots)
        p4_targets = self._rank_high_value_singles(fishes, current_score)
        ranked.extend(p4_targets)
        
        # ===== TIER 5: MANDATORY FIRE (30s+) =====
        # Only if must fire: pick highest-confidence remaining target
        if must_fire_soon:
            p5_targets = self._rank_mandatory_fire(fishes, current_score)
            ranked.extend(p5_targets)
        
        # Sort by confidence * expected_value
        ranked = sorted(ranked, key=lambda x: -(x['confidence'] * x['expected_gain']))
        
        # CRITICAL: Filter out low-confidence or negative-value shots
        if not must_fire_soon:
            ranked = [t for t in ranked if t['confidence'] >= 0.5 and t['expected_gain'] >= 0]
        else:
            # Even in mandatory mode, only include targets with >40% confidence
            ranked = [t for t in ranked if t['confidence'] >= 0.4]
        
        return ranked, self.fire_aggression, must_fire_soon
    
    def _rank_high_confidence_killshots(self, fishes, current_score):
        """
        TIER 1: Fish classes with 65%+ success history.
        These are PROVEN. Fire without hesitation.
        """
        p1_targets = []
        
        for fish in fishes:
            # Check if we have high confidence data
            confidence = self.shot_tracker.get_confidence_for_class(
                fish.class_id, 
                min_threshold=65
            )
            
            if confidence > 0:
                value = self._estimate_value(fish.area)
                burst_cost = 3
                expected_gain = value - burst_cost
                
                # Add a small bonus for high confidence
                if confidence > 0.75:
                    expected_gain *= 1.1
                
                # Only fire if break-even or better
                if expected_gain >= 0:
                    p1_targets.append({
                        'fish': fish,
                        'type': 'killshot_proven',
                        'confidence': confidence,
                        'burst_target': self._calculate_lead(fish),
                        'burst_size': 3,
                        'expected_gain': expected_gain,
                        'efficiency': value / burst_cost,
                        'priority_tier': 1
                    })
        
        return p1_targets
    
    def _rank_convergence_confidence(self, fishes, region, current_score):
        """
        TIER 2: 2+ targets converging geometrically.
        Require 65%+ geometric confidence.
        Only fire if both targets + convergence point = positive value.
        """
        p2_targets = []
        
        small_fish = [f for f in fishes if f.class_id <= 1 and f.speed > 1.5]
        
        if len(small_fish) < 2:
            return p2_targets
        
        for i in range(len(small_fish)):
            for j in range(i + 1, len(small_fish)):
                f1, f2 = small_fish[i], small_fish[j]
                
                convergence_data = self._predict_convergence(f1, f2, region)
                
                # STRICT: Only if 65%+ convergence confidence
                if convergence_data and convergence_data['confidence'] >= 0.65:
                    value_f1 = self._estimate_value(f1.area)
                    value_f2 = self._estimate_value(f2.area)
                    total_value = value_f1 + value_f2
                    
                    burst_cost = 3
                    expected_gain = total_value - burst_cost
                    
                    # Only fire if we gain value
                    if expected_gain > 0:
                        for fish in [f1, f2]:
                            p2_targets.append({
                                'fish': fish,
                                'type': 'convergence_high_confidence',
                                'confidence': convergence_data['confidence'],
                                'burst_target': (
                                    int(convergence_data['x']),
                                    int(convergence_data['y'])
                                ),
                                'burst_size': 3,
                                'expected_gain': expected_gain,
                                'efficiency': total_value / burst_cost,
                                'priority_tier': 2
                            })
        
        return p2_targets
    
    def _rank_cluster_efficiency(self, fishes, current_score):
        """
        TIER 3: Clusters of 3+ small targets.
        Fire if efficiency > 1.0 (more value than cost).
        """
        p3_targets = []
        
        # Find tight clusters of small fish
        small_fish = [f for f in fishes if f.class_id <= 1]
        
        processed = set()
        for i, fish in enumerate(small_fish):
            if fish.id in processed:
                continue
            
            # Find all fish within cluster distance
            cluster = [f for f in small_fish if 
                      abs(f.x - fish.x) < 80 and abs(f.y - fish.y) < 80]
            
            # STRICT: Only 3+ clusters
            if len(cluster) >= 3:
                cluster_value = sum([self._estimate_value(f.area) for f in cluster])
                burst_cost = 2  # Efficient
                efficiency = cluster_value / burst_cost
                expected_gain = cluster_value - burst_cost
                
                # Only fire if efficiency > 1.0
                if efficiency > 1.0 and expected_gain > 0:
                    for f in cluster:
                        if f.id not in processed:
                            # Lead to cluster center
                            cluster_center_x = np.mean([c.x for c in cluster])
                            cluster_center_y = np.mean([c.y for c in cluster])
                            
                            p3_targets.append({
                                'fish': f,
                                'type': 'cluster_3plus',
                                'confidence': 0.7 + (0.1 * (len(cluster) - 3)),  # More confident with larger clusters
                                'burst_target': (int(cluster_center_x), int(cluster_center_y)),
                                'burst_size': 2,
                                'expected_gain': expected_gain / len(cluster),
                                'efficiency': efficiency,
                                'priority_tier': 3
                            })
                            processed.add(f.id)
        
        return p3_targets
    
    def _rank_high_value_singles(self, fishes, current_score):
        """
        TIER 4: Single large targets where value >> cost.
        Only fire if value >= 2x total shot cost.
        Conservative but profitable.
        """
        p4_targets = []
        
        for fish in fishes:
            if fish.class_id >= 2:  # Large only
                value = self._estimate_value(fish.area)
                hp = self._estimate_hp(fish.area)
                shot_cost = 2
                total_cost = hp * shot_cost
                
                # STRICT: Value must be 2x cost
                profit = value - total_cost
                
                if profit > value * 0.3:  # At least 30% profit margin
                    p4_targets.append({
                        'fish': fish,
                        'type': 'single_large_profitable',
                        'confidence': 0.6,  # Moderate confidence on large targets
                        'burst_target': self._calculate_lead(fish),
                        'burst_size': 2,
                        'expected_gain': profit,
                        'efficiency': value / total_cost,
                        'priority_tier': 4
                    })
        
        return p4_targets
    
    def _rank_mandatory_fire(self, fishes, current_score):
        """
        TIER 5: 30+ seconds idle.
        Pick highest-confidence target only.
        NEVER fire on random marginal targets.
        """
        p5_targets = []
        
        # Get all targets with >40% confidence
        candidates = []
        
        for fish in fishes:
            # Check historical confidence
            hist_confidence = self.shot_tracker.get_confidence_for_class(
                fish.class_id, 
                min_threshold=40  # Lower threshold for mandatory
            )
            
            # Calculate basic value
            value = self._estimate_value(fish.area)
            
            if value >= 1:  # At least minimal value
                confidence = max(hist_confidence, 0.4)  # Default 40% if no history
                expected_gain = value - 1
                
                candidates.append({
                    'fish': fish,
                    'type': 'mandatory_fire_backup',
                    'confidence': confidence,
                    'burst_target': self._calculate_lead(fish),
                    'burst_size': 1,
                    'expected_gain': expected_gain,
                    'efficiency': value / 1,
                    'priority_tier': 5
                })
        
        # Sort by confidence, pick top
        candidates = sorted(candidates, key=lambda x: -x['confidence'])
        
        if candidates:
            p5_targets.append(candidates[0])
        
        return p5_targets
    
    def _predict_convergence(self, f1, f2, region):
        """
        Geometric convergence prediction.
        Returns confidence 0-1 based on trajectory likelihood.
        """
        x1, y1, vx1, vy1 = f1.x, f1.y, f1.vx, f1.vy
        x2, y2, vx2, vy2 = f2.x, f2.y, f2.vx, f2.vy
        
        dx = x2 - x1
        dy = y2 - y1
        initial_dist = np.sqrt(dx**2 + dy**2)
        
        # Project forward to find closest approach
        min_dist = initial_dist
        convergence_point = None
        
        for frame in range(1, 25):  # 25 frames = ~0.8 seconds
            proj_x1 = x1 + vx1 * frame
            proj_y1 = y1 + vy1 * frame
            proj_x2 = x2 + vx2 * frame
            proj_y2 = y2 + vy2 * frame
            
            dist = np.sqrt((proj_x2 - proj_x1)**2 + (proj_y2 - proj_y1)**2)
            
            if dist < min_dist:
                min_dist = dist
                convergence_point = ((proj_x1 + proj_x2) / 2, (proj_y1 + proj_y2) / 2)
        
        # Confidence = how much distance decreases
        if min_dist < initial_dist * 0.7:  # 30%+ closer = converging
            confidence = 1.0 - (min_dist / initial_dist)
            return {
                'x': convergence_point[0],
                'y': convergence_point[1],
                'confidence': max(0, min(1, confidence)),
                'distance_reduction': initial_dist - min_dist
            }
        
        return None
    
    def _calculate_lead(self, fish):
        """Calculate lead point for moving target"""
        lead_frames = 5
        lead_x = int(fish.x + fish.vx * lead_frames)
        lead_y = int(fish.y + fish.vy * lead_frames)
        return (lead_x, lead_y)
    
    def record_shot_fired(self, value_gained, shots_in_burst):
        """Track shot efficiency"""
        self.last_shot_time = time.time()
        self.fire_aggression = 0
        self.total_value_gained += value_gained
        self.total_shots_fired += shots_in_burst
        
        if value_gained < 0:
            self.wasted_shots += shots_in_burst
        
        if self.total_shots_fired > 0:
            self.efficiency_ratio = self.total_value_gained / self.total_shots_fired
    
    def get_fire_urgency(self):
        """0=normal, 1=prep, 2=must fire"""
        time_since = time.time() - self.last_shot_time
        
        if time_since > self.mandatory_fire_interval:
            return 2
        elif time_since > self.mandatory_fire_interval * 0.67:
            return 1
        else:
            return 0
    
    def get_stats(self):
        return {
            'value_gained': self.total_value_gained,
            'shots_fired': self.total_shots_fired,
            'efficiency': self.efficiency_ratio,
            'wasted_shots': self.wasted_shots,
            'waste_rate': self.wasted_shots / max(self.total_shots_fired, 1)
        }
    
    def _estimate_value(self, area):
        if area < 150: return 2
        elif area < 400: return 5
        else: return 15
    
    def _estimate_hp(self, area):
        if area < 150: return 1
        elif area < 400: return 3
        else: return 6