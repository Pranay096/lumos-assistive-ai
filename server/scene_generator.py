import random
import string

class SceneGenerator:
    @staticmethod
    def generate(task_id: str) -> dict:
        if task_id == "blind_mode":
            return SceneGenerator.generate_blind_mode()
        elif task_id == "deaf_mode":
            return SceneGenerator.generate_deaf_mode()
        elif task_id == "mute_mode":
            return SceneGenerator.generate_mute_mode()
        raise ValueError(f"Unknown task_id: {task_id}")

    @staticmethod
    def generate_blind_mode() -> dict:
        contexts = [
            "User navigating a dimly lit {location} looking for the {destination}.",
            "User walking through a cluttered {location} attempting to reach the {destination}.",
            "User trying to find their way out of a noisy {location} towards the {destination}.",
            "User crossing a busy {location} to locate the {destination}."
        ]
        locations = ["marketplace", "hospital ward", "underground parking lot", "basement storage", "kitchen", "office hallway", "train station", "subway terminal", "pharmacy aisle", "supermarket"]
        destinations = ["exit route", "consultation room", "medication cabinet", "checkout counter", "ticket booth", "elevator bank", "emergency stairwell"]

        hazard_pools = [
            "wet_floor_patch", "open_drawer", "staircase_edge", "electrical_cable", 
            "biohazard_bin", "wet_floor_sign", "scattered_tools", "spilled_chemicals", 
            "unmarked_step", "abandoned_cart", "glass_shards", "construction_debris",
            "puddle_of_water", "toppled_trashcan"
        ]
        
        object_pools = [
            "fire_extinguisher", "junction_box", "medical_cart", "room_number_plate",
            "wheelchair_ramp", "storage_boxes", "market_stall", "bicycle", "pharmacy_sign",
            "hallway", "medication_cabinet", "crowd", "reception_desk", "vending_machine",
            "waiting_chairs", "info_kiosk", "trash_receptacle", "potted_plant", "delivery_box"
        ]

        # Combinatorial sampling
        num_hazards = random.randint(1, 2)
        hazards = random.sample(hazard_pools, num_hazards)
        
        num_objects = random.randint(3, 6)
        safe_objects = random.sample([obj for obj in object_pools if obj not in hazards], num_objects)
        
        all_objects = hazards + safe_objects
        random.shuffle(all_objects)

        # Picking a sign
        text_object = random.choice(safe_objects) if safe_objects else "signboard"
        if text_object not in all_objects:
            all_objects.append(text_object)
            
        sign_templates = [
            "RxCare Pharmacy — {time} — {info}",
            "Room {room_num} — {department} — {info}",
            "EMERGENCY EXIT → {info}",
            "Take {dose} {drug}. {info}",
            "NOTICE: {info}"
        ]
        
        times = ["Open 9am–9pm", "Closed for lunch", "24/7 Hours"]
        infos = ["Do NOT use elevator", "Knock before entering", "Avoid alcohol", "Follow green arrows", "Next refill: 14 Apr", "Staff only beyond this point"]
        deps = ["Oncology Consultation", "Radiology", "Pediatrics", "Cardiology"]
        drugs = ["Metformin", "Ibuprofen", "Amoxicillin", "Lisinopril"]
        
        sign_text = random.choice(sign_templates).format(
            time=random.choice(times),
            info=random.choice(infos),
            room_num=f"{random.randint(1, 9)}{random.choice(['A', 'B', 'C', 'D'])}",
            department=random.choice(deps),
            dose=f"{random.randint(10, 500)}mg",
            drug=random.choice(drugs)
        )

        user_context_str = random.choice(contexts).format(
            location=random.choice(locations),
            destination=random.choice(destinations)
        )

        # 25% chance of a hardware interrupt
        interrupt = None
        if random.random() < 0.25:
            interrupt = {
                "type": random.choice(["new_hazard", "user_command", "bluetooth_ping"]),
                "content": f"System Alert: {random.choice(['Smoke detected', 'User asks for directions', 'Battery low', 'Signal dropped'])}",
                "urgency": random.choice(["high", "critical", "low"])
            }

        return {
            "user_context": user_context_str,
            "all_objects": all_objects,
            "hazards": hazards,
            "text_objects": [text_object],
            "text_content": {text_object: sign_text},
            "interrupt_event": interrupt,
            "failure_mode": random.choice([None, None, "dark", "heavy_noise"]) if random.random() < 0.3 else None
        }

    @staticmethod
    def generate_deaf_mode() -> dict:
        templates = [
            "The {patient_type} presented with {symptom} requiring {drug_dosage} of {medication} every {time_interval}. Please monitor for {side_effect}.",
            "We have a {patient_type} suffering from {symptom}. Administer {medication} immediately, {drug_dosage} IV. Watch for {side_effect}.",
            "Code blue in {location}. {patient_type} unresponsive due to {symptom}. Push {drug_dosage} {medication} now. Next check in {time_interval}.",
            "Routine checkup for {patient_type}. History of {symptom}. Continuing {medication} at {drug_dosage}. Review again in {time_interval} to check for {side_effect}."
        ]
        
        patient_types = ["elderly male", "pediatric patient", "pregnant female", "adult male", "adolescent female", "trauma victim", "post-op patient"]
        symptoms = ["acute myocardial infarction", "anaphylactic shock", "pulmonary embolism", "diabetic ketoacidosis", "severe sepsis", "status epilepticus", "hypertensive crisis", "ventricular fibrillation", "cerebral hemorrhage"]
        medications = ["epinephrine", "amiodarone", "norepinephrine", "dexamethasone", "naloxone", "adenosine", "heparin", "fentanyl", "propofol", "ketamine", "vancomycin", "ceftriaxone"]
        dosages = ["1mg", "0.5mg", "150mg", "300mg", "50mcg", "2mg/kg", "10 units", "1 liter", "0.1mg/kg"]
        intervals = ["two hours", "four hours", "six hours", "twelve hours", "fifteen minutes", "thirty minutes"]
        side_effects = ["tachycardia", "bradycardia", "hypotension", "respiratory depression", "allergic reaction", "arrhythmia"]
        locations = ["ICU", "Emergency Room", "Ward 4B", "Triage", "Operating Theater 2"]

        target_med = random.choice(medications)
        target_symptom = random.choice(symptoms)
        target_dosage = random.choice(dosages)
        target_interval = random.choice(intervals)

        transcript = random.choice(templates).format(
            patient_type=random.choice(patient_types),
            symptom=target_symptom,
            medication=target_med,
            drug_dosage=target_dosage,
            time_interval=target_interval,
            side_effect=random.choice(side_effects),
            location=random.choice(locations)
        )

        key_terms = [target_med, target_symptom, target_dosage, target_interval]

        interrupt = None
        if random.random() < 0.25:
             interrupt = {
                "type": "audio_interference",
                "content": "Loud PA system announcement overlapping with speaker.",
                "urgency": random.choice(["low", "high"])
             }

        return {
            "user_context": "Listening to a clinical hand-off report.",
            "full_transcript": transcript,
            "key_terms": key_terms,
            "interrupt_event": interrupt,
            "failure_mode": random.choice([None, None, "heavy_noise"]) if random.random() < 0.3 else None
        }

    @staticmethod
    def generate_mute_mode() -> dict:
        # Instead of picking from a fixed dictionary, generate a random phonetic-like sequence.
        # This absolutely scales the difficulty and combinatorics to infinity.
        length = random.randint(3, 5)
        vowels = "AEIOU"
        consonants = "BCDFGHJKLMNPQRSTVWXYZ"
        
        # Build a procedural word
        word = ""
        for i in range(length):
            if i % 2 == 0:
                word += random.choice(consonants)
            else:
                word += random.choice(vowels)
                
        # Occasionally mix it up
        word_list = list(word)
        random.shuffle(word_list)
        word = "".join(word_list)

        return {
            "target_word": word,
            "frames": list(word),
            "interrupt_event": None,
            "failure_mode": random.choice([None, None, "heavy_noise"]) if random.random() < 0.3 else None
        }
