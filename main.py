import cv2
import numpy as np
import mediapipe as mp

class HandGesturePainter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
        self.current_color_index = 1  # Start with Green
        self.last_point = None
        self.gesture_cooldown = 0
        self.eraser_mode = False
        self.close_gesture_frames = 0

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get hand landmarks
                    h, w, _ = frame.shape
                    landmarks = self.get_landmark_coordinates(hand_landmarks, w, h)
                    
                    # Recognize gestures and perform actions
                    if self.recognize_close_gesture(landmarks):
                        self.close_gesture_frames += 1
                        if self.close_gesture_frames > 30:  # Hold for 1 second (assuming 30 fps)
                            print("Closing application...")
                            return
                    else:
                        self.close_gesture_frames = 0
                    
                    self.recognize_gestures(landmarks)
                    
                    # Draw or erase based on the gesture
                    self.draw_or_erase(landmarks)
            else:
                self.close_gesture_frames = 0

            # Overlay drawing on frame
            frame = cv2.addWeighted(frame, 1, self.drawing_canvas, 0.5, 0)
            
            # Display color selection and eraser options
            self.display_options(frame)

            cv2.imshow('Hand Gesture Painter', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def get_landmark_coordinates(self, landmarks, width, height):
        return {
            name: (int(landmark.x * width), int(landmark.y * height))
            for name, landmark in enumerate(landmarks.landmark)
        }

    def recognize_gestures(self, landmarks):
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return

        # Change color: Thumb, Index, and Middle fingers up
        if (self.is_finger_up(landmarks, self.mp_hands.HandLandmark.THUMB_TIP) and
            self.is_finger_up(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP) and
            self.is_finger_up(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP) and
            not self.is_finger_up(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP)):
            self.current_color_index = (self.current_color_index + 1) % len(self.colors)
            self.gesture_cooldown = 15  # Prevent rapid changes

        # Check for palm facing camera (eraser mode)
        self.eraser_mode = self.is_palm_facing_camera(landmarks)

    def recognize_close_gesture(self, landmarks):
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Check if middle finger is up and other fingers are down
        return (middle_tip[1] < middle_pip[1] and
                ring_tip[1] > middle_pip[1] and
                pinky_tip[1] > middle_pip[1])

    def is_finger_up(self, landmarks, finger_tip):
        return landmarks[finger_tip][1] < landmarks[self.mp_hands.HandLandmark.WRIST][1]

    def is_palm_facing_camera(self, landmarks):
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        middle_finger_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_finger_mcp = landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        
        # Check if the middle and ring finger MCPs are to the right of the wrist
        return (middle_finger_mcp[0] > wrist[0] and ring_finger_mcp[0] > wrist[0])

    def draw_or_erase(self, landmarks):
        index_finger_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        if self.eraser_mode:
            # Use palm center for erasing
            palm_center = self.calculate_palm_center(landmarks)
            if self.last_point:
                cv2.line(self.drawing_canvas, self.last_point, palm_center, (0, 0, 0), 40)
            self.last_point = palm_center
        else:
            # Draw with index finger
            if self.is_drawing_gesture(landmarks):
                if self.last_point:
                    cv2.line(self.drawing_canvas, self.last_point, index_finger_tip, self.colors[self.current_color_index], 5)
                self.last_point = index_finger_tip
            else:
                self.last_point = None

    def is_drawing_gesture(self, landmarks):
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        return self.calculate_distance(thumb_tip, index_tip) < 40

    def calculate_palm_center(self, landmarks):
        palm_landmarks = [
            self.mp_hands.HandLandmark.WRIST,
            self.mp_hands.HandLandmark.THUMB_CMC,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        x = sum(landmarks[lm][0] for lm in palm_landmarks) // len(palm_landmarks)
        y = sum(landmarks[lm][1] for lm in palm_landmarks) // len(palm_landmarks)
        return (x, y)

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def display_options(self, frame):
        # Display current color/mode
        color = (200, 200, 200) if self.eraser_mode else self.colors[self.current_color_index]
        cv2.circle(frame, (30, 30), 20, color, -1)
        cv2.putText(frame, "Eraser" if self.eraser_mode else "Draw", (60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

if __name__ == "__main__":
    painter = HandGesturePainter()
    painter.run()