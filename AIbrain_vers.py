import numpy as np
import random
import copy
import string


class AIbrain_vers:
    """
    Ensemble AI mozek (verze 9 - ANTI-FREEZE):

    Hlavní vylepšení:
    1. SILNÝ anti-freeze mechanismus - auta nikdy nezamrznou
    2. Penalizace za stání ve fitness funkci
    3. Zjednodušená logika brzdění
    4. Zachována racing line logika z v8
    """

    def __init__(self):
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0

        self.x = 0
        self.y = 0
        self.speed = 0

        self.history_length = 5
        self.sensor_history = []

        self.init_param()

    def init_param(self):
        """Inicializace obou modelů."""
        self.sensors_per_frame = 9

        self.input_size_A = (self.sensors_per_frame + 1) * self.history_length + self.sensors_per_frame
        self.hidden_size_A = 32
        self.output_size_A = 4

        self.W1_A = np.random.randn(self.input_size_A, self.hidden_size_A) * 0.3
        self.b1_A = np.zeros(self.hidden_size_A)
        self.W2_A = np.random.randn(self.hidden_size_A, self.output_size_A) * 0.3
        self.b2_A = np.zeros(self.output_size_A)

        # Silnější bias pro jízdu vpřed
        self.b2_A[0] = 0.7  # Plyn (zvýšeno)
        self.b2_A[1] = -0.5  # Brzda (sníženo)

        self.input_size_B = self.input_size_A + self.output_size_A
        self.hidden_size_B = 24
        self.output_size_B = self.sensors_per_frame

        self.W1_B = np.random.randn(self.input_size_B, self.hidden_size_B) * 0.3
        self.b1_B = np.zeros(self.hidden_size_B)
        self.W2_B = np.random.randn(self.hidden_size_B, self.output_size_B) * 0.3
        self.b2_B = np.zeros(self.output_size_B)

        self.centering_acc = 0.0
        neutral_frame = np.array([3.0] * self.sensors_per_frame + [0.0])
        self.sensor_history = [neutral_frame.copy() for _ in range(self.history_length)]
        self.prev_sensors = np.zeros(self.sensors_per_frame)

        # Detekce startu
        self.start_x = None
        self.start_y = None
        self.passed_checkpoint = False
        self.completed_laps = 0
        self.total_distance = 0.0

        # Racing line
        self.racing_line_bonus = 0.0
        self.last_situation = "straight"

        # Anti-freeze tracking
        self.freeze_frames = 0  # Počet framů s nízkou rychlostí
        self.low_speed_penalty = 0.0

        self.NAME = "VERS_Ensemble"
        self.store()

    def detect_situation(self, sensors):
        """Detekce situace na trati."""
        if len(sensors) < 9:
            return "straight", 0.0

        left_90 = sensors[0]
        left_45 = sensors[2]
        right_45 = sensors[6]
        right_90 = sensors[8]

        left_avg = (left_90 + left_45) / 2
        right_avg = (right_90 + right_45) / 2
        asymmetry = left_avg - right_avg

        TURN_THRESHOLD = 0.8

        if abs(asymmetry) < TURN_THRESHOLD:
            return "straight", asymmetry
        elif asymmetry > 0:
            return "right_turn", asymmetry
        else:
            return "left_turn", asymmetry

    def calculate_racing_line_bonus(self, sensors, situation):
        """Zjednodušený racing line bonus."""
        if len(sensors) < 9:
            return 0.0

        left_90 = sensors[0]
        right_90 = sensors[8]

        bonus = 0.0

        if situation == "right_turn":
            # Zatáčka doprava - vnitřní strana je VPRAVO
            if 0.8 <= right_90 <= 1.8:
                bonus = 1.0
            elif 0.5 <= right_90 <= 2.5:
                bonus = 0.5

        elif situation == "left_turn":
            # Zatáčka doleva - vnitřní strana je VLEVO
            if 0.8 <= left_90 <= 1.8:
                bonus = 1.0
            elif 0.5 <= left_90 <= 2.5:
                bonus = 0.5

        elif situation == "straight":
            # Na rovince bonus za centrování
            if abs(left_90 - right_90) < 1.0:
                bonus = 0.3

        return bonus

    def decide(self, data):
        """Ensemble rozhodování s ANTI-FREEZE."""
        self.decider += 1
        sensors = np.array(data, dtype=float).ravel()
        self.last_sensor_data = sensors

        # Detekce situace
        situation, asymmetry = self.detect_situation(sensors)
        self.last_situation = situation

        # Racing line bonus
        rl_bonus = self.calculate_racing_line_bonus(sensors, situation)
        self.racing_line_bonus += rl_bonus * 0.01

        # Neural network processing
        normalized_speed = self.speed / 500.0
        current_frame = np.append(sensors[:self.sensors_per_frame], normalized_speed)

        sensor_derivatives = sensors[:self.sensors_per_frame] - self.prev_sensors
        self.prev_sensors = sensors[:self.sensors_per_frame].copy()

        self.sensor_history.pop(0)
        self.sensor_history.append(current_frame)

        history_flat = np.concatenate(self.sensor_history)
        input_A = np.concatenate([history_flat, sensor_derivatives])

        if input_A.size < self.input_size_A:
            input_A = np.concatenate([input_A, np.zeros(self.input_size_A - input_A.size)])
        elif input_A.size > self.input_size_A:
            input_A = input_A[:self.input_size_A]

        h_A = np.tanh(np.dot(input_A, self.W1_A) + self.b1_A)
        logits_A = np.dot(h_A, self.W2_A) + self.b2_A
        action_A = 1.0 / (1.0 + np.exp(-logits_A))

        input_B = np.concatenate([input_A, action_A])

        if input_B.size < self.input_size_B:
            input_B = np.concatenate([input_B, np.zeros(self.input_size_B - input_B.size)])
        elif input_B.size > self.input_size_B:
            input_B = input_B[:self.input_size_B]

        h_B = np.tanh(np.dot(input_B, self.W1_B) + self.b1_B)
        predicted_sensors = np.dot(h_B, self.W2_B) + self.b2_B

        final_action = action_A.copy()

        # === ANTI-FREEZE SYSTÉM ===
        front_sensor = sensors[4] if len(sensors) > 4 else 3.0

        # Sledování zamrznutí
        if self.speed < 50:
            self.freeze_frames += 1
        else:
            self.freeze_frames = max(0, self.freeze_frames - 2)

        # LEVEL 1: Nízká rychlost + volno vpředu
        if self.speed < 100 and front_sensor > 1.2:
            final_action[0] = max(final_action[0], 0.75)
            final_action[1] = min(final_action[1], 0.15)

        # LEVEL 2: Velmi nízká rychlost
        if self.speed < 50 and front_sensor > 0.6:
            final_action[0] = max(final_action[0], 0.9)
            final_action[1] = min(final_action[1], 0.1)

        # LEVEL 3: Zamrznutí detekováno - NOUZOVÝ REŽIM
        if self.freeze_frames > 10:
            final_action[0] = 1.0  # Plný plyn
            final_action[1] = 0.0  # Žádná brzda
            # Přidat náhodné zatáčení pro vyjetí ze stuck pozice
            if self.freeze_frames > 20:
                if random.random() > 0.5:
                    final_action[2] = max(final_action[2], 0.7)  # Vlevo
                else:
                    final_action[3] = max(final_action[3], 0.7)  # Vpravo

        # Korekce při nebezpečí - VELMI MÍRNÁ
        predicted_front = predicted_sensors[4] if len(predicted_sensors) > 4 else 3.0

        if predicted_front < 0.8:  # Jen při opravdovém nebezpečí
            danger_level = 1.0 - (predicted_front / 0.8)
            final_action[1] = min(0.5, final_action[1] + danger_level * 0.2)
            final_action[0] = max(0.3, final_action[0] - danger_level * 0.1)

        return final_action

    def calculate_score(self, distance, time, no):
        """Fitness s penalizací za zamrznutí."""
        # Reset na začátku nové epochy
        if distance < self.total_distance * 0.5 or distance < 1.0:
            self.completed_laps = 0
            self.passed_checkpoint = False
            self.centering_acc = 0.0
            self.racing_line_bonus = 0.0
            self.freeze_frames = 0
            self.low_speed_penalty = 0.0

        self.total_distance = distance

        # Penalizace za nízkou rychlost
        if self.speed < 50:
            self.low_speed_penalty += 0.1

        # === SKÓRE ===
        base_score = distance * 5.0
        lap_bonus = self.completed_laps * 200.0
        racing_bonus = self.racing_line_bonus * 60.0

        # Efektivita
        efficiency_bonus = 0.0
        if self.completed_laps > 0 and time > 0:
            time_per_lap = time / self.completed_laps
            if time_per_lap < 10.0:
                efficiency_bonus = (10.0 - time_per_lap) * 25.0 * self.completed_laps

        # Rychlost bonus
        avg_speed = distance / time if time > 0 else 0
        speed_bonus = avg_speed * 1.5  # Zvýšeno pro motivaci k rychlosti

        # Penalizace za zamrznutí
        freeze_penalty = self.low_speed_penalty * 2.0

        self.score = base_score + lap_bonus + racing_bonus + efficiency_bonus + speed_bonus - freeze_penalty

    def mutate(self):
        """Mutace obou modelů."""
        mutation_rate = 0.25

        self.W1_A += np.random.randn(*self.W1_A.shape) * mutation_rate
        self.b1_A += np.random.randn(*self.b1_A.shape) * mutation_rate
        self.W2_A += np.random.randn(*self.W2_A.shape) * mutation_rate
        self.b2_A += np.random.randn(*self.b2_A.shape) * mutation_rate

        self.W1_B += np.random.randn(*self.W1_B.shape) * mutation_rate
        self.b1_B += np.random.randn(*self.b1_B.shape) * mutation_rate
        self.W2_B += np.random.randn(*self.W2_B.shape) * mutation_rate
        self.b2_B += np.random.randn(*self.b2_B.shape) * mutation_rate

        self.NAME += "m"
        self.store()

    def store(self):
        """Uložení všech parametrů."""
        self.parameters = copy.deepcopy({
            "W1_A": self.W1_A, "b1_A": self.b1_A,
            "W2_A": self.W2_A, "b2_A": self.b2_A,
            "W1_B": self.W1_B, "b1_B": self.b1_B,
            "W2_B": self.W2_B, "b2_B": self.b2_B,
            "NAME": self.NAME,
        })

    def set_parameters(self, parameters):
        """Načtení parametrů."""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        try:
            loaded_W1_A = np.array(params_dict["W1_A"], dtype=float)
            if loaded_W1_A.shape != self.W1_A.shape:
                print(f"POZOR: Nekompatibilní mozek. Ignoruji.")
                return
        except Exception as e:
            print(f"Chyba: {e}")
            return

        self.parameters = params_dict

        self.W1_A = np.array(self.parameters["W1_A"], dtype=float)
        self.b1_A = np.array(self.parameters["b1_A"], dtype=float)
        self.W2_A = np.array(self.parameters["W2_A"], dtype=float)
        self.b2_A = np.array(self.parameters["b2_A"], dtype=float)

        self.W1_B = np.array(self.parameters["W1_B"], dtype=float)
        self.b1_B = np.array(self.parameters["b1_B"], dtype=float)
        self.W2_B = np.array(self.parameters["W2_B"], dtype=float)
        self.b2_B = np.array(self.parameters["b2_B"], dtype=float)

        self.NAME = str(self.parameters["NAME"])

        # Reset
        self.passed_checkpoint = False
        self.completed_laps = 0
        self.total_distance = 0.0
        self.centering_acc = 0.0
        self.racing_line_bonus = 0.0
        self.freeze_frames = 0
        self.low_speed_penalty = 0.0

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

        if self.start_x is None:
            self.start_x = x
            self.start_y = y

        checkpoint_y = self.start_y - 400
        far_from_start_x = abs(x - self.start_x) > 160

        if y < checkpoint_y and far_from_start_x:
            self.passed_checkpoint = True

        tolerance = 100
        in_start_zone = abs(x - self.start_x) < tolerance and abs(y - self.start_y) < tolerance

        if in_start_zone and self.passed_checkpoint and self.total_distance > 35.0:
            self.completed_laps += 1
            self.passed_checkpoint = False
            print(f"[{self.NAME}] Kolo {self.completed_laps}!")

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)
