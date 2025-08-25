# Agentes Inteligentes: Fundamentos e Implementações Avançadas

## Introdução aos Agentes Inteligentes

Um agente inteligente é uma entidade computacional autônoma que percebe seu ambiente através de sensores, processa informações usando sua base de conhecimento, toma decisões baseadas em seus objetivos, e age sobre o ambiente através de atuadores para maximizar sua função de utilidade.

## Arquitetura Fundamental de Agentes

### Modelo PEAS (Performance, Environment, Actuators, Sensors)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

class Agent(ABC):
    """Classe base abstrata para agentes inteligentes"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.performance_measure = 0
        self.knowledge_base = {}
        self.internal_state = {}
        self.action_history = []
        self.perception_history = []
    
    @abstractmethod
    def perceive(self, environment: 'Environment') -> Dict[str, Any]:
        """Percepção do ambiente através de sensores"""
        pass
    
    @abstractmethod
    def think(self, percepts: Dict[str, Any]) -> str:
        """Processamento cognitivo e tomada de decisão"""
        pass
    
    @abstractmethod
    def act(self, action: str, environment: 'Environment') -> bool:
        """Execução de ações no ambiente"""
        pass
    
    def agent_program(self, environment: 'Environment') -> str:
        """Programa principal do agente (ciclo percepção-ação)"""
        # Percepção
        percepts = self.perceive(environment)
        self.perception_history.append(percepts)
        
        # Atualização do estado interno
        self.update_internal_state(percepts)
        
        # Tomada de decisão
        action = self.think(percepts)
        
        # Execução da ação
        success = self.act(action, environment)
        
        # Registro da ação
        self.action_history.append({
            'action': action,
            'success': success,
            'timestamp': time.time()
        })
        
        return action
    
    def update_internal_state(self, percepts: Dict[str, Any]):
        """Atualiza estado interno baseado nas percepções"""
        self.internal_state.update({
            'last_percepts': percepts,
            'step_count': len(self.perception_history),
            'last_update': time.time()
        })
    
    def evaluate_performance(self) -> float:
        """Avalia performance do agente"""
        return self.performance_measure
```

## Taxonomia de Agentes Inteligentes

### 1. Agentes Reativos Simples

```python
class SimpleReflexAgent(Agent):
    """Agente que age baseado apenas no estado atual"""
    
    def __init__(self, agent_id: str, rules: Dict[str, str]):
        super().__init__(agent_id)
        self.condition_action_rules = rules
        
    def perceive(self, environment: 'Environment') -> Dict[str, Any]:
        """Percepção direta do estado atual"""
        current_state = environment.get_current_state()
        return {
            'current_location': current_state.get('agent_location'),
            'visible_objects': current_state.get('visible_objects', []),
            'local_conditions': current_state.get('local_conditions', {})
        }
    
    def think(self, percepts: Dict[str, Any]) -> str:
        """Seleção de ação baseada em regras condição-ação"""
        for condition, action in self.condition_action_rules.items():
            if self.evaluate_condition(condition, percepts):
                return action
        
        return 'no_action'
    
    def evaluate_condition(self, condition: str, percepts: Dict[str, Any]) -> bool:
        """Avalia se uma condição é verdadeira"""
        # Implementação simplificada - em caso real, usaria parser
        if condition == 'obstacle_ahead':
            return 'obstacle' in percepts.get('visible_objects', [])
        elif condition == 'goal_visible':
            return 'goal' in percepts.get('visible_objects', [])
        elif condition == 'battery_low':
            return percepts.get('local_conditions', {}).get('battery_level', 100) < 20
        
        return False
    
    def act(self, action: str, environment: 'Environment') -> bool:
        """Executa ação no ambiente"""
        return environment.execute_action(self.agent_id, action)

# Exemplo de uso
robot_rules = {
    'obstacle_ahead': 'turn_right',
    'goal_visible': 'move_toward_goal',
    'battery_low': 'return_to_base',
    'default': 'explore'
}

reflex_robot = SimpleReflexAgent('robot_1', robot_rules)
```

### 2. Agentes Reativos Baseados em Modelo

```python
class ModelBasedReflexAgent(Agent):
    """Agente que mantém modelo interno do mundo"""
    
    def __init__(self, agent_id: str, world_model: 'WorldModel'):
        super().__init__(agent_id)
        self.world_model = world_model
        self.internal_world_state = {}
        
    def perceive(self, environment: 'Environment') -> Dict[str, Any]:
        """Percepção com integração ao modelo do mundo"""
        raw_percepts = environment.get_percepts_for_agent(self.agent_id)
        
        return {
            'sensor_data': raw_percepts,
            'derived_information': self.world_model.interpret_percepts(raw_percepts),
            'confidence_levels': self.world_model.get_confidence_levels()
        }
    
    def think(self, percepts: Dict[str, Any]) -> str:
        """Tomada de decisão baseada no modelo do mundo"""
        # Atualizar modelo do mundo
        self.world_model.update(percepts, self.action_history)
        
        # Predizer estados futuros
        predicted_states = self.world_model.predict_future_states(steps=3)
        
        # Avaliar ações possíveis
        possible_actions = self.world_model.get_possible_actions()
        action_evaluations = {}
        
        for action in possible_actions:
            expected_outcome = self.world_model.predict_action_outcome(action)
            utility = self.calculate_utility(expected_outcome)
            action_evaluations[action] = utility
        
        # Selecionar melhor ação
        best_action = max(action_evaluations, key=action_evaluations.get)
        return best_action
    
    def calculate_utility(self, outcome: Dict[str, Any]) -> float:
        """Calcula utilidade de um resultado esperado"""
        utility = 0.0
        
        # Fatores de utilidade
        if outcome.get('goal_reached'):
            utility += 100
        
        if outcome.get('obstacle_avoided'):
            utility += 50
        
        if outcome.get('energy_consumed', 0) > 0:
            utility -= outcome['energy_consumed'] * 0.1
        
        if outcome.get('risk_level', 0) > 0:
            utility -= outcome['risk_level'] * 20
        
        return utility
    
    def act(self, action: str, environment: 'Environment') -> bool:
        """Executa ação e atualiza modelo"""
        success = environment.execute_action(self.agent_id, action)
        
        # Atualizar modelo baseado no resultado
        actual_outcome = environment.get_action_result(self.agent_id, action)
        self.world_model.update_with_outcome(action, actual_outcome, success)
        
        return success

class WorldModel:
    """Modelo do mundo para agente baseado em modelo"""
    
    def __init__(self):
        self.world_state = {}
        self.transition_model = {}  # Como ações mudam o estado
        self.sensor_model = {}      # Como o estado gera percepções
        self.uncertainty_levels = {}
        
    def update(self, percepts: Dict, action_history: List):
        """Atualiza modelo com novas percepções"""
        # Filtragem bayesiana ou Kalman para atualizar crenças
        for key, value in percepts['derived_information'].items():
            if key in self.world_state:
                # Combinar informação anterior com nova
                old_value = self.world_state[key]
                confidence = percepts['confidence_levels'].get(key, 0.8)
                
                # Atualização bayesiana simplificada
                self.world_state[key] = (
                    old_value * (1 - confidence) + value * confidence
                )
            else:
                self.world_state[key] = value
    
    def predict_future_states(self, steps: int) -> List[Dict]:
        """Prediz estados futuros baseado no modelo de transição"""
        future_states = []
        current_state = self.world_state.copy()
        
        for step in range(steps):
            # Aplicar modelo de transição
            next_state = self.apply_transition_model(current_state)
            future_states.append(next_state)
            current_state = next_state
        
        return future_states
    
    def predict_action_outcome(self, action: str) -> Dict[str, Any]:
        """Prediz resultado de uma ação"""
        if action in self.transition_model:
            outcome = self.transition_model[action](self.world_state)
            return outcome
        
        return {'unknown_action': True}
```

### 3. Agentes Baseados em Objetivos

```python
from queue import PriorityQueue
import heapq

class GoalBasedAgent(Agent):
    """Agente que planeja ações para alcançar objetivos"""
    
    def __init__(self, agent_id: str, goals: List[str], planner: 'Planner'):
        super().__init__(agent_id)
        self.goals = goals
        self.current_plan = []
        self.planner = planner
        self.goal_priorities = {goal: 1.0 for goal in goals}
        
    def perceive(self, environment: 'Environment') -> Dict[str, Any]:
        """Percepção orientada a objetivos"""
        raw_percepts = environment.get_full_state()
        
        return {
            'world_state': raw_percepts,
            'goal_progress': self.evaluate_goal_progress(raw_percepts),
            'obstacles_to_goals': self.identify_obstacles(raw_percepts),
            'opportunities': self.identify_opportunities(raw_percepts)
        }
    
    def think(self, percepts: Dict[str, Any]) -> str:
        """Planejamento baseado em objetivos"""
        world_state = percepts['world_state']
        
        # Verificar se plano atual ainda é válido
        if not self.is_plan_valid(world_state):
            # Replanejar
            self.current_plan = self.create_new_plan(world_state)
        
        # Executar próximo passo do plano
        if self.current_plan:
            next_action = self.current_plan.pop(0)
            return next_action
        
        return 'no_action'
    
    def create_new_plan(self, world_state: Dict) -> List[str]:
        """Cria novo plano para alcançar objetivos"""
        # Selecionar objetivo com maior prioridade
        current_goal = self.select_current_goal(world_state)
        
        if current_goal:
            # Usar algoritmo de planejamento (A*, STRIPS, etc.)
            plan = self.planner.plan(
                initial_state=world_state,
                goal=current_goal,
                available_actions=self.get_available_actions()
            )
            return plan
        
        return []
    
    def select_current_goal(self, world_state: Dict) -> Optional[str]:
        """Seleciona objetivo atual baseado em prioridades e viabilidade"""
        viable_goals = []
        
        for goal in self.goals:
            if not self.is_goal_achieved(goal, world_state):
                viability = self.assess_goal_viability(goal, world_state)
                priority = self.goal_priorities[goal]
                score = priority * viability
                viable_goals.append((score, goal))
        
        if viable_goals:
            viable_goals.sort(reverse=True)
            return viable_goals[0][1]
        
        return None
    
    def is_goal_achieved(self, goal: str, world_state: Dict) -> bool:
        """Verifica se objetivo foi alcançado"""
        # Implementação específica do domínio
        goal_conditions = self.parse_goal(goal)
        return all(self.check_condition(cond, world_state) for cond in goal_conditions)
    
    def act(self, action: str, environment: 'Environment') -> bool:
        """Executa ação e monitora progresso dos objetivos"""
        success = environment.execute_action(self.agent_id, action)
        
        # Atualizar progresso dos objetivos
        new_state = environment.get_full_state()
        self.update_goal_progress(new_state)
        
        return success

class AStarPlanner:
    """Planejador usando algoritmo A*"""
    
    def __init__(self):
        self.heuristics = {}
        
    def plan(self, initial_state: Dict, goal: str, available_actions: List[str]) -> List[str]:
        """Planejamento usando A*"""
        open_set = PriorityQueue()
        open_set.put((0, id(initial_state), initial_state, []))
        
        closed_set = set()
        
        while not open_set.empty():
            f_score, _, current_state, path = open_set.get()
            
            state_key = self.state_to_key(current_state)
            if state_key in closed_set:
                continue
            
            closed_set.add(state_key)
            
            # Verificar se objetivo foi alcançado
            if self.is_goal_state(current_state, goal):
                return path
            
            # Expandir estados sucessores
            for action in available_actions:
                next_state = self.apply_action(current_state, action)
                next_state_key = self.state_to_key(next_state)
                
                if next_state_key not in closed_set:
                    new_path = path + [action]
                    g_score = len(new_path)
                    h_score = self.heuristic(next_state, goal)
                    f_score = g_score + h_score
                    
                    open_set.put((f_score, id(next_state), next_state, new_path))
        
        return []  # Não encontrou plano
    
    def heuristic(self, state: Dict, goal: str) -> float:
        """Função heurística para A*"""
        # Implementação específica do domínio
        # Exemplo: distância Manhattan para navegação
        if goal == 'reach_location':
            agent_pos = state.get('agent_position', (0, 0))
            goal_pos = state.get('goal_position', (0, 0))
            return abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        
        return 0.0
    
    def apply_action(self, state: Dict, action: str) -> Dict:
        """Aplica ação ao estado e retorna novo estado"""
        new_state = state.copy()
        
        # Implementação específica das ações
        if action == 'move_north':
            pos = new_state.get('agent_position', (0, 0))
            new_state['agent_position'] = (pos[0], pos[1] + 1)
        elif action == 'move_south':
            pos = new_state.get('agent_position', (0, 0))
            new_state['agent_position'] = (pos[0], pos[1] - 1)
        # ... outras ações
        
        return new_state
```

### 4. Agentes Baseados em Utilidade

```python
class UtilityBasedAgent(Agent):
    """Agente que maximiza função de utilidade"""
    
    def __init__(self, agent_id: str, utility_function: 'UtilityFunction'):
        super().__init__(agent_id)
        self.utility_function = utility_function
        self.action_values = {}
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        
    def think(self, percepts: Dict[str, Any]) -> str:
        """Seleção de ação baseada em maximização de utilidade"""
        current_state = percepts['world_state']
        available_actions = self.get_available_actions(current_state)
        
        # Calcular utilidade esperada para cada ação
        action_utilities = {}
        
        for action in available_actions:
            expected_utility = self.calculate_expected_utility(current_state, action)
            action_utilities[action] = expected_utility
        
        # Estratégia epsilon-greedy para exploração
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        # Selecionar ação com maior utilidade esperada
        best_action = max(action_utilities, key=action_utilities.get)
        return best_action
    
    def calculate_expected_utility(self, state: Dict, action: str) -> float:
        """Calcula utilidade esperada de uma ação"""
        possible_outcomes = self.predict_outcomes(state, action)
        expected_utility = 0.0
        
        for outcome, probability in possible_outcomes:
            utility = self.utility_function.evaluate(outcome)
            expected_utility += probability * utility
        
        return expected_utility
    
    def predict_outcomes(self, state: Dict, action: str) -> List[tuple]:
        """Prediz possíveis resultados de uma ação com probabilidades"""
        # Modelo estocástico de transição
        outcomes = []
        
        if action == 'move_forward':
            # 80% chance de sucesso, 10% falha, 10% efeito colateral
            success_state = self.apply_action(state, action)
            fail_state = state.copy()  # Não se move
            side_effect_state = self.apply_side_effect(state, action)
            
            outcomes = [
                (success_state, 0.8),
                (fail_state, 0.1),
                (side_effect_state, 0.1)
            ]
        
        return outcomes
    
    def learn_from_experience(self, state: Dict, action: str, outcome: Dict, reward: float):
        """Aprendizado baseado em experiência"""
        # Atualizar estimativas de utilidade
        state_action_key = (self.state_to_key(state), action)
        
        if state_action_key not in self.action_values:
            self.action_values[state_action_key] = 0.0
        
        # Atualização temporal da diferença
        old_value = self.action_values[state_action_key]
        self.action_values[state_action_key] = (
            old_value + self.learning_rate * (reward - old_value)
        )
        
        # Atualizar função de utilidade
        self.utility_function.update(outcome, reward)

class UtilityFunction:
    """Função de utilidade multi-objetivo"""
    
    def __init__(self, objectives: Dict[str, float]):
        self.objectives = objectives  # {objetivo: peso}
        self.normalization_factors = {}
        
    def evaluate(self, state: Dict) -> float:
        """Avalia utilidade de um estado"""
        total_utility = 0.0
        
        for objective, weight in self.objectives.items():
            objective_value = self.evaluate_objective(objective, state)
            normalized_value = self.normalize_value(objective, objective_value)
            total_utility += weight * normalized_value
        
        return total_utility
    
    def evaluate_objective(self, objective: str, state: Dict) -> float:
        """Avalia um objetivo específico"""
        if objective == 'goal_distance':
            agent_pos = state.get('agent_position', (0, 0))
            goal_pos = state.get('goal_position', (10, 10))
            distance = np.sqrt((agent_pos[0] - goal_pos[0])**2 + (agent_pos[1] - goal_pos[1])**2)
            return -distance  # Menor distância = maior utilidade
        
        elif objective == 'energy_conservation':
            return state.get('energy_level', 100)
        
        elif objective == 'safety':
            danger_level = state.get('danger_level', 0)
            return 100 - danger_level
        
        elif objective == 'exploration':
            visited_cells = len(state.get('visited_locations', set()))
            return visited_cells
        
        return 0.0
    
    def normalize_value(self, objective: str, value: float) -> float:
        """Normaliza valor do objetivo para [0, 1]"""
        if objective not in self.normalization_factors:
            return value
        
        min_val, max_val = self.normalization_factors[objective]
        return (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0
```

### 5. Agentes de Aprendizado

```python
class LearningAgent(Agent):
    """Agente que aprende através da experiência"""
    
    def __init__(self, agent_id: str, learning_algorithm: str = 'q_learning'):
        super().__init__(agent_id)
        self.learning_algorithm = learning_algorithm
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        self.experience_buffer = []
        
    def think(self, percepts: Dict[str, Any]) -> str:
        """Seleção de ação com aprendizado"""
        state = self.extract_state_features(percepts)
        available_actions = self.get_available_actions(state)
        
        if self.learning_algorithm == 'q_learning':
            return self.q_learning_action_selection(state, available_actions)
        elif self.learning_algorithm == 'policy_gradient':
            return self.policy_gradient_action_selection(state, available_actions)
        elif self.learning_algorithm == 'actor_critic':
            return self.actor_critic_action_selection(state, available_actions)
        
        return random.choice(available_actions)
    
    def q_learning_action_selection(self, state: tuple, available_actions: List[str]) -> str:
        """Seleção de ação usando Q-Learning"""
        # Inicializar Q-values se necessário
        for action in available_actions:
            if (state, action) not in self.q_table:
                self.q_table[(state, action)] = 0.0
        
        # Estratégia epsilon-greedy
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        # Selecionar ação com maior Q-value
        q_values = [self.q_table[(state, action)] for action in available_actions]
        max_q = max(q_values)
        best_actions = [action for action in available_actions 
                       if self.q_table[(state, action)] == max_q]
        
        return random.choice(best_actions)
    
    def update_q_values(self, state: tuple, action: str, reward: float, next_state: tuple):
        """Atualização Q-Learning"""
        current_q = self.q_table.get((state, action), 0.0)
        
        # Calcular max Q-value para próximo estado
        next_actions = self.get_available_actions(next_state)
        if next_actions:
            max_next_q = max(self.q_table.get((next_state, a), 0.0) for a in next_actions)
        else:
            max_next_q = 0.0
        
        # Atualização Q-Learning
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state, action)] = new_q
    
    def learn_from_experience(self, experience: Dict):
        """Aprendizado a partir da experiência"""
        state = experience['state']
        action = experience['action']
        reward = experience['reward']
        next_state = experience['next_state']
        done = experience['done']
        
        if self.learning_algorithm == 'q_learning':
            self.update_q_values(state, action, reward, next_state)
        
        # Armazenar experiência para replay
        self.experience_buffer.append(experience)
        
        # Manter buffer limitado
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)
        
        # Decaimento da exploração
        self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
    
    def experience_replay(self, batch_size: int = 32):
        """Replay de experiências para aprendizado"""
        if len(self.experience_buffer) < batch_size:
            return
        
        batch = random.sample(self.experience_buffer, batch_size)
        
        for experience in batch:
            self.learn_from_experience(experience)

class DeepQLearningAgent(LearningAgent):
    """Agente com Deep Q-Learning"""
    
    def __init__(self, agent_id: str, state_size: int, action_size: int):
        super().__init__(agent_id, 'deep_q_learning')
        self.state_size = state_size
        self.action_size = action_size
        
        # Redes neurais
        self.q_network = self.build_q_network()
        self.target_network = self.build_q_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        self.update_target_frequency = 100
        self.training_step = 0
        
    def build_q_network(self) -> nn.Module:
        """Constrói rede neural para Q-values"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
    
    def deep_q_action_selection(self, state: np.ndarray) -> int:
        """Seleção de ação usando Deep Q-Network"""
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def train_deep_q_network(self, batch: List[Dict]):
        """Treinamento da rede neural"""
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        # Atualizar rede target
        if self.training_step % self.update_target_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
```

## Sistemas Multi-Agentes

### Comunicação e Coordenação

```python
class MultiAgentSystem:
    """Sistema multi-agente com comunicação e coordenação"""
    
    def __init__(self):
        self.agents = {}
        self.communication_network = {}
        self.shared_knowledge = {}
        self.coordination_mechanisms = []
        
    def add_agent(self, agent: Agent):
        """Adiciona agente ao sistema"""
        self.agents[agent.agent_id] = agent
        self.communication_network[agent.agent_id] = []
        
    def establish_communication(self, agent1_id: str, agent2_id: str):
        """Estabelece canal de comunicação entre agentes"""
        if agent1_id in self.communication_network:
            self.communication_network[agent1_id].append(agent2_id)
        if agent2_id in self.communication_network:
            self.communication_network[agent2_id].append(agent1_id)
    
    def broadcast_message(self, sender_id: str, message: Dict, recipients: List[str] = None):
        """Transmite mensagem entre agentes"""
        if recipients is None:
            recipients = self.communication_network.get(sender_id, [])
        
        for recipient_id in recipients:
            if recipient_id in self.agents:
                self.agents[recipient_id].receive_message(sender_id, message)
    
    def coordinate_actions(self, coordination_type: str = 'consensus'):
        """Coordena ações entre agentes"""
        if coordination_type == 'consensus':
            return self.consensus_coordination()
        elif coordination_type == 'auction':
            return self.auction_coordination()
        elif coordination_type == 'hierarchical':
            return self.hierarchical_coordination()
    
    def consensus_coordination(self) -> Dict[str, str]:
        """Coordenação por consenso"""
        agent_proposals = {}
        
        # Fase 1: Coleta de propostas
        for agent_id, agent in self.agents.items():
            proposal = agent.propose_action()
            agent_proposals[agent_id] = proposal
        
        # Fase 2: Negociação
        rounds = 0
        max_rounds = 10
        
        while rounds < max_rounds:
            conflicts = self.detect_conflicts(agent_proposals)
            if not conflicts:
                break
            
            # Resolver conflitos
            for conflict in conflicts:
                resolution = self.resolve_conflict(conflict, agent_proposals)
                agent_proposals.update(resolution)
            
            rounds += 1
        
        return agent_proposals
    
    def auction_coordination(self) -> Dict[str, str]:
        """Coordenação por leilão"""
        tasks = self.identify_tasks()
        assignments = {}
        
        for task in tasks:
            bids = {}
            
            # Coleta de lances
            for agent_id, agent in self.agents.items():
                if agent.can_perform_task(task):
                    bid = agent.bid_for_task(task)
                    bids[agent_id] = bid
            
            # Selecionar vencedor
            if bids:
                winner = max(bids, key=bids.get)
                assignments[winner] = task
                
                # Notificar resultado
                self.broadcast_message('system', {
                    'type': 'task_assignment',
                    'task': task,
                    'assigned_to': winner,
                    'winning_bid': bids[winner]
                })
        
        return assignments

class CommunicativeAgent(Agent):
    """Agente com capacidades de comunicação"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.inbox = []
        self.communication_protocols = {}
        self.negotiation_history = []
        
    def receive_message(self, sender_id: str, message: Dict):
        """Recebe mensagem de outro agente"""
        self.inbox.append({
            'sender': sender_id,
            'message': message,
            'timestamp': time.time()
        })
    
    def send_message(self, recipient_id: str, message: Dict, system: MultiAgentSystem):
        """Envia mensagem para outro agente"""
        system.broadcast_message(self.agent_id, message, [recipient_id])
    
    def process_messages(self):
        """Processa mensagens recebidas"""
        for message_data in self.inbox:
            message = message_data['message']
            sender = message_data['sender']
            
            if message['type'] == 'information_sharing':
                self.process_information(message['content'])
            elif message['type'] == 'negotiation_proposal':
                self.process_negotiation_proposal(sender, message)
            elif message['type'] == 'coordination_request':
                self.process_coordination_request(sender, message)
        
        self.inbox.clear()
    
    def negotiate_with_agent(self, other_agent_id: str, issue: Dict, system: MultiAgentSystem):
        """Negocia com outro agente"""
        proposal = self.generate_proposal(issue)
        
        message = {
            'type': 'negotiation_proposal',
            'issue': issue,
            'proposal': proposal,
            'negotiation_id': f"{self.agent_id}_{other_agent_id}_{time.time()}"
        }
        
        self.send_message(other_agent_id, message, system)
        self.negotiation_history.append(message)
    
    def generate_proposal(self, issue: Dict) -> Dict:
        """Gera proposta para negociação"""
        # Implementação específica do domínio
        return {
            'terms': issue.get('terms', {}),
            'concessions': self.calculate_concessions(issue),
            'deadline': time.time() + 3600  # 1 hora
        }
```

## Aplicações Práticas

### Agente de Trading Financeiro

```python
class TradingAgent(UtilityBasedAgent):
    """Agente de trading financeiro"""
    
    def __init__(self, agent_id: str, portfolio: Dict, risk_tolerance: float):
        utility_function = TradingUtilityFunction(risk_tolerance)
        super().__init__(agent_id, utility_function)
        self.portfolio = portfolio
        self.market_data = {}
        self.trading_history = []
        self.risk_tolerance = risk_tolerance
        
    def perceive(self, environment: 'MarketEnvironment') -> Dict[str, Any]:
        """Percepção dos dados de mercado"""
        return {
            'market_prices': environment.get_current_prices(),
            'market_indicators': environment.get_technical_indicators(),
            'news_sentiment': environment.get_news_sentiment(),
            'portfolio_value': self.calculate_portfolio_value(environment),
            'market_volatility': environment.get_volatility_measures()
        }
    
    def calculate_expected_utility(self, state: Dict, action: str) -> float:
        """Calcula utilidade esperada de operação de trading"""
        if action.startswith('buy_'):
            asset = action.split('_')[1]
            return self.calculate_buy_utility(asset, state)
        elif action.startswith('sell_'):
            asset = action.split('_')[1]
            return self.calculate_sell_utility(asset, state)
        elif action == 'hold':
            return self.calculate_hold_utility(state)
        
        return 0.0
    
    def calculate_buy_utility(self, asset: str, state: Dict) -> float:
        """Calcula utilidade de compra"""
        current_price = state['market_prices'][asset]
        predicted_price = self.predict_price(asset, state)
        
        expected_return = (predicted_price - current_price) / current_price
        risk = state['market_volatility'][asset]
        
        # Função de utilidade com aversão ao risco
        utility = expected_return - (self.risk_tolerance * risk ** 2)
        
        return utility
    
    def predict_price(self, asset: str, state: Dict) -> float:
        """Predição de preço usando indicadores técnicos"""
        current_price = state['market_prices'][asset]
        indicators = state['market_indicators'][asset]
        
        # Modelo simplificado - em prática usaria ML
        trend_factor = indicators.get('moving_average_trend', 1.0)
        momentum_factor = indicators.get('rsi_momentum', 1.0)
        sentiment_factor = state['news_sentiment'].get(asset, 1.0)
        
        predicted_price = current_price * trend_factor * momentum_factor * sentiment_factor
        
        return predicted_price

class TradingUtilityFunction(UtilityFunction):
    """Função de utilidade específica para trading"""
    
    def __init__(self, risk_tolerance: float):
        objectives = {
            'profit': 0.6,
            'risk_management': 0.3,
            'diversification': 0.1
        }
        super().__init__(objectives)
        self.risk_tolerance = risk_tolerance
    
    def evaluate_objective(self, objective: str, state: Dict) -> float:
        if objective == 'profit':
            return state.get('portfolio_return', 0.0)
        elif objective == 'risk_management':
            portfolio_risk = state.get('portfolio_risk', 0.0)
            return max(0, 1.0 - portfolio_risk / self.risk_tolerance)
        elif objective == 'diversification':
            return state.get('diversification_score', 0.0)
        
        return 0.0
```

### Agente de Recomendação Inteligente

```python
class RecommendationAgent(LearningAgent):
    """Agente de sistema de recomendação"""
    
    def __init__(self, agent_id: str, user_profiles: Dict, item_catalog: Dict):
        super().__init__(agent_id, 'collaborative_filtering')
        self.user_profiles = user_profiles
        self.item_catalog = item_catalog
        self.interaction_matrix = {}
        self.similarity_cache = {}
        
    def perceive(self, environment: 'RecommendationEnvironment') -> Dict[str, Any]:
        """Percepção de interações e contexto do usuário"""
        return {
            'user_interactions': environment.get_recent_interactions(),
            'user_context': environment.get_user_context(),
            'item_popularity': environment.get_item_popularity(),
            'seasonal_trends': environment.get_seasonal_trends(),
            'real_time_events': environment.get_real_time_events()
        }
    
    def generate_recommendations(self, user_id: str, num_recommendations: int = 10) -> List[Dict]:
        """Gera recomendações personalizadas"""
        user_profile = self.user_profiles.get(user_id, {})
        
        # Múltiplas estratégias de recomendação
        collaborative_recs = self.collaborative_filtering_recommendations(user_id)
        content_based_recs = self.content_based_recommendations(user_id)
        popularity_recs = self.popularity_based_recommendations()
        
        # Combinar recomendações
        combined_recs = self.combine_recommendations([
            (collaborative_recs, 0.5),
            (content_based_recs, 0.3),
            (popularity_recs, 0.2)
        ])
        
        # Aplicar filtros e diversificação
        filtered_recs = self.apply_filters(combined_recs, user_profile)
        diversified_recs = self.diversify_recommendations(filtered_recs)
        
        return diversified_recs[:num_recommendations]
    
    def collaborative_filtering_recommendations(self, user_id: str) -> List[Dict]:
        """Recomendações por filtragem colaborativa"""
        similar_users = self.find_similar_users(user_id)
        recommendations = []
        
        for similar_user, similarity in similar_users:
            user_items = self.get_user_items(similar_user)
            target_user_items = self.get_user_items(user_id)
            
            for item_id, rating in user_items.items():
                if item_id not in target_user_items:
                    predicted_rating = rating * similarity
                    recommendations.append({
                        'item_id': item_id,
                        'predicted_rating': predicted_rating,
                        'recommendation_type': 'collaborative'
                    })
        
        return sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
    
    def content_based_recommendations(self, user_id: str) -> List[Dict]:
        """Recomendações baseadas em conteúdo"""
        user_profile = self.user_profiles[user_id]
        user_preferences = user_profile.get('preferences', {})
        
        recommendations = []
        
        for item_id, item_data in self.item_catalog.items():
            if not self.user_has_interacted(user_id, item_id):
                similarity = self.calculate_content_similarity(user_preferences, item_data)
                recommendations.append({
                    'item_id': item_id,
                    'predicted_rating': similarity,
                    'recommendation_type': 'content_based'
                })
        
        return sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
    
    def learn_from_feedback(self, user_id: str, item_id: str, feedback: float):
        """Aprende com feedback do usuário"""
        # Atualizar matriz de interação
        if user_id not in self.interaction_matrix:
            self.interaction_matrix[user_id] = {}
        
        self.interaction_matrix[user_id][item_id] = feedback
        
        # Atualizar perfil do usuário
        self.update_user_profile(user_id, item_id, feedback)
        
        # Invalidar cache de similaridade
        if user_id in self.similarity_cache:
            del self.similarity_cache[user_id]
        
        # Aprendizado online
        experience = {
            'state': self.get_recommendation_state(user_id),
            'action': f'recommend_{item_id}',
            'reward': feedback,
            'next_state': self.get_recommendation_state(user_id),
            'done': False
        }
        
        self.learn_from_experience(experience)
```

## Conclusão

Os agentes inteligentes representam uma abordagem fundamental para criar sistemas autônomos capazes de operar em ambientes complexos e dinâmicos. Esta análise abrangente demonstrou:

### **Arquiteturas Principais**:
1. **Agentes Reativos**: Resposta direta a estímulos
2. **Agentes Baseados em Modelo**: Representação interna do mundo
3. **Agentes Orientados a Objetivos**: Planejamento para alcançar metas
4. **Agentes Baseados em Utilidade**: Maximização de funções de valor
5. **Agentes de Aprendizado**: Adaptação através da experiência

### **Características Avançadas**:
- **Comunicação Multi-Agente**: Coordenação e colaboração
- **Aprendizado Contínuo**: Melhoria da performance ao longo do tempo
- **Tomada de Decisão Sob Incerteza**: Lidar com ambientes estocásticos
- **Planejamento Hierárquico**: Decomposição de objetivos complexos

### **Aplicações Práticas**:
- **Trading Financeiro**: Análise de mercado e tomada de decisão
- **Sistemas de Recomendação**: Personalização e filtragem
- **Robótica**: Navegação e manipulação autônoma
- **Jogos**: NPCs inteligentes e estratégias adaptativas

O futuro dos agentes inteligentes aponta para sistemas cada vez mais sofisticados, com capacidades de raciocínio causal, aprendizado few-shot e colaboração natural com humanos. A integração com Large Language Models e técnicas de aprendizado por reforço profundo promete criar agentes verdadeiramente autônomos e versáteis.
