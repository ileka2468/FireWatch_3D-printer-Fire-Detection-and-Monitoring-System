import cv2
import time
import configparser

def get_ender_config():
    config = configparser.ConfigParser()
    config.read("dist/FireWatch/firedetection.ini")
    
    # Extract Ender 3 settings
    for section in config.sections():
        if "Ender" in section:
            return {
                "url": config.get(section, "url"),
                "zoom": config.getfloat(section, "zoom", fallback=1.0),
                "pan_x": config.getfloat(section, "pan_x", fallback=0.0),
                "pan_y": config.getfloat(section, "pan_y", fallback=0.0)
            }
    return None

def apply_crop(frame, zoom, pan_x, pan_y):
    if zoom <= 1.01:
        return frame
        
    h, w = frame.shape[:2]
    new_w = int(w / zoom)
    new_h = int(h / zoom)
    
    base_x = (w - new_w) // 2
    base_y = (h - new_h) // 2
    
    offset_x = int(base_x * pan_x)
    offset_y = int(base_y * pan_y)
    
    x1 = max(0, min(w - new_w, base_x + offset_x))
    y1 = max(0, min(h - new_h, base_y + offset_y))
    
    return frame[y1:y1+new_h, x1:x1+new_w]

def main():
    cfg = get_ender_config()
    if not cfg:
        print("Could not find Ender config in INI file.")
        return
        
    print(f"Connecting to: {cfg['url']}")
    cap = cv2.VideoCapture(cfg['url'])
    
    if not cap.isOpened():
        print("Error opening stream.")
        return
        
    history = []
    
    cv2.namedWindow("1. Cropped Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. Motion Mask (512x512)", cv2.WINDOW_NORMAL)
    
    last_check_time = time.time()
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed.")
            time.sleep(1)
            cap = cv2.VideoCapture(cfg['url'])
            continue
            
        cropped = apply_crop(frame, cfg['zoom'], cfg['pan_x'], cfg['pan_y'])
        cv2.imshow("1. Cropped Frame", cropped)
        
        now = time.time()
        
        if now - last_check_time >= 1.0:
            last_check_time = now
            
            # --- Exactly what the NEW engine does ---
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (512, 512))
            small_gray = cv2.GaussianBlur(small_gray, (3, 3), 0)
            
            history = [h for h in history if now - h[0] < 15]
            
            motion_detected = False
            max_changed = 0
            best_mask = None
            
            for _, old_frame in history:
                diff = cv2.absdiff(small_gray, old_frame)
                _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
                changed_pixels = cv2.countNonZero(thresh)
                
                if changed_pixels > max_changed:
                    max_changed = changed_pixels
                    best_mask = thresh
                    
                if changed_pixels > 50:
                    motion_detected = True
            
            if best_mask is not None:
                display_mask = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
                color = (0, 0, 255) if motion_detected else (0, 255, 0)
                status = "MOTION" if motion_detected else "STATIC"
                
                cv2.putText(display_mask, f"Max Changed: {max_changed} (Thresh: 50)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display_mask, f"Status: {status}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("2. Motion Mask (512x512)", display_mask)
            
            history.append((now, small_gray))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
