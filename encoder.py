#!/usr/bin/env python3
"""
MT6701 Magnetic Encoder - AB Mode (Optimized Polling)
Raspberry Pi 5 GPIO Configuration

Wiring:
- VCC: 3.3V or 5V
- GND: Ground
- A: GPIO 17 (Phase A output)
- B: GPIO 27 (Phase B output) 
"""

import RPi.GPIO as GPIO
import time
import threading
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class EncoderState:
    position: int = 0
    direction: int = 0  # 1 = CW, -1 = CCW, 0 = stopped
    zero_offset: int = 0
    velocity: float = 0.0  # counts per second
    rpm: float = 0.0
    acceleration: float = 0.0  # counts per second squared
    error_count: int = 0
    poll_count: int = 0

class MT6701Encoder:
    # Encoder resolution (4x quadrature = 4096 counts per revolution)
    COUNTS_PER_REVOLUTION = 4096
    
    # Optimized lookup table for quadrature decoding
    # Index = (last_state << 2) | current_state
    DECODER_TABLE = [
        0,   # 0000: 00 -> 00 (no change)
        -1,  # 0001: 00 -> 01 (CCW)
        1,   # 0010: 00 -> 10 (CW)
        2,   # 0011: 00 -> 11 (error)
        1,   # 0100: 01 -> 00 (CW)
        0,   # 0101: 01 -> 01 (no change)
        2,   # 0110: 01 -> 10 (error)
        -1,  # 0111: 01 -> 11 (CCW)
        -1,  # 1000: 10 -> 00 (CCW)
        2,   # 1001: 10 -> 01 (error)
        0,   # 1010: 10 -> 10 (no change)
        1,   # 1011: 10 -> 11 (CW)
        2,   # 1100: 11 -> 00 (error)
        1,   # 1101: 11 -> 01 (CW)
        -1,  # 1110: 11 -> 10 (CCW)
        0    # 1111: 11 -> 11 (no change)
    ]

    def __init__(self, a_pin=17, b_pin=27, poll_rate_hz=20000):
        self.a_pin = a_pin
        self.b_pin = b_pin
        self.poll_rate_hz = poll_rate_hz
        self.poll_period = 1.0 / poll_rate_hz
        
        self.encoder_state = EncoderState()
        self.last_encoded = 0
        self.running = True
        self.state_lock = threading.Lock()
        
        # Velocity tracking
        self.position_history = deque(maxlen=50)
        self.velocity_history = deque(maxlen=20)
        self.last_velocity_time = time.perf_counter()
        
        # Performance monitoring
        self.poll_times = deque(maxlen=1000)
        self.missed_polls = 0
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.a_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.b_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Get initial state
        self.last_encoded = (GPIO.input(self.a_pin) << 1) | GPIO.input(self.b_pin)
        
        # Start high-priority polling thread
        self.poll_thread = threading.Thread(target=self._optimized_poll_encoder, daemon=True)
        self.poll_thread.start()
        
        # Start velocity calculation thread
        self.velocity_thread = threading.Thread(target=self._calculate_velocity, daemon=True)
        self.velocity_thread.start()
        
        # Zero the encoder
        self.zero_position()
        print(f"Encoder initialized (optimized polling at {poll_rate_hz}Hz)")

    def _optimized_poll_encoder(self):
        """Optimized polling with minimal overhead"""
        # Pre-calculate for efficiency
        next_poll_time = time.perf_counter()
        
        # Local references for speed
        a_pin = self.a_pin
        b_pin = self.b_pin
        decoder = self.DECODER_TABLE
        gpio_input = GPIO.input
        
        while self.running:
            current_time = time.perf_counter()
            
            # Track if we're keeping up with desired poll rate
            if current_time > next_poll_time + self.poll_period:
                self.missed_polls += 1
            
            # Read both pins in quick succession
            encoded = (gpio_input(a_pin) << 1) | gpio_input(b_pin)
            
            # Decode using lookup table
            index = (self.last_encoded << 2) | encoded
            change = decoder[index]
            
            if change == 2:  # Error condition
                with self.state_lock:
                    self.encoder_state.error_count += 1
            elif change != 0:  # Valid movement
                with self.state_lock:
                    self.encoder_state.position += change
                    self.encoder_state.direction = change
                    # Store position with timestamp for velocity calculation
                    self.position_history.append((current_time, self.encoder_state.position))
            
            self.last_encoded = encoded
            
            # Performance tracking
            self.poll_times.append(current_time)
            with self.state_lock:
                self.encoder_state.poll_count += 1
            
            # Calculate next poll time to maintain consistent rate
            next_poll_time += self.poll_period
            
            # Smart sleep - only if we have time
            sleep_time = next_poll_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _calculate_velocity(self):
        """Calculate velocity, RPM, and acceleration in separate thread"""
        while self.running:
            current_time = time.perf_counter()
            
            with self.state_lock:
                if len(self.position_history) >= 10:
                    # Use linear regression for smooth velocity estimation
                    times = np.array([p[0] for p in list(self.position_history)[-10:]])
                    positions = np.array([p[1] for p in list(self.position_history)[-10:]])
                    
                    # Normalize time to avoid numerical issues
                    times = times - times[0]
                    
                    if times[-1] > 0:
                        # Linear fit: position = velocity * time + offset
                        velocity = np.polyfit(times, positions, 1)[0]
                        self.encoder_state.velocity = velocity
                        self.encoder_state.rpm = (velocity / self.COUNTS_PER_REVOLUTION) * 60
                        
                        # Store velocity history for acceleration
                        self.velocity_history.append((current_time, velocity))
                        
                        # Calculate acceleration if we have enough velocity data
                        if len(self.velocity_history) >= 5:
                            v_times = np.array([v[0] for v in list(self.velocity_history)[-5:]])
                            velocities = np.array([v[1] for v in list(self.velocity_history)[-5:]])
                            v_times = v_times - v_times[0]
                            
                            if v_times[-1] > 0:
                                self.encoder_state.acceleration = np.polyfit(v_times, velocities, 1)[0]
                
                # Check if stopped (no new positions in last 0.1 seconds)
                if (len(self.position_history) == 0 or 
                    current_time - self.position_history[-1][0] > 0.1):
                    self.encoder_state.velocity = 0
                    self.encoder_state.rpm = 0
                    self.encoder_state.acceleration = 0
                    self.encoder_state.direction = 0
            
            time.sleep(0.02)  # Update 50 times per second

    def get_position(self):
        """Get raw encoder position (counts)"""
        with self.state_lock:
            return self.encoder_state.position - self.encoder_state.zero_offset

    def get_position_degrees(self):
        """Get current encoder position in degrees"""
        position = self.get_position()
        return (position / self.COUNTS_PER_REVOLUTION) * 360.0

    def get_position_radians(self):
        """Get current encoder position in radians"""
        position = self.get_position()
        return (position / self.COUNTS_PER_REVOLUTION) * 2 * 3.14159265359

    def get_velocity(self):
        """Get current velocity in counts per second"""
        with self.state_lock:
            return self.encoder_state.velocity

    def get_rpm(self):
        """Get current RPM"""
        with self.state_lock:
            return self.encoder_state.rpm

    def get_acceleration(self):
        """Get current acceleration in counts per second squared"""
        with self.state_lock:
            return self.encoder_state.acceleration

    def get_direction(self):
        """Get current rotation direction"""
        with self.state_lock:
            return self.encoder_state.direction

    def get_performance_stats(self):
        """Get performance statistics"""
        with self.state_lock:
            if len(self.poll_times) >= 2:
                actual_rate = len(self.poll_times) / (self.poll_times[-1] - self.poll_times[0])
            else:
                actual_rate = 0
            
            return {
                'target_poll_rate': self.poll_rate_hz,
                'actual_poll_rate': actual_rate,
                'poll_efficiency': (actual_rate / self.poll_rate_hz * 100) if self.poll_rate_hz > 0 else 0,
                'total_polls': self.encoder_state.poll_count,
                'missed_polls': self.missed_polls,
                'error_count': self.encoder_state.error_count,
                'error_rate': (self.encoder_state.error_count / self.encoder_state.poll_count * 100) if self.encoder_state.poll_count > 0 else 0
            }

    def zero_position(self):
        """Set current position as zero"""
        with self.state_lock:
            self.encoder_state.zero_offset = self.encoder_state.position
            self.position_history.clear()
            self.velocity_history.clear()

    def set_poll_rate(self, rate_hz):
        """Dynamically adjust polling rate"""
        self.poll_rate_hz = rate_hz
        self.poll_period = 1.0 / rate_hz

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.poll_thread.is_alive():
            self.poll_thread.join(timeout=1.0)
        if self.velocity_thread.is_alive():
            self.velocity_thread.join(timeout=1.0)
        GPIO.cleanup()

def main():
    """Example usage with enhanced display"""
    print("MT6701 Encoder Test - Optimized Polling Method")
    print("GPIO Configuration:")
    print("  A: GPIO 17")
    print("  B: GPIO 27")
    print("-" * 70)
    
    # Create encoder with 20kHz polling rate
    encoder = MT6701Encoder(poll_rate_hz=20000)
    
    try:
        print("Monitoring encoder... (Ctrl+C to stop)")
        print("\nPress 'z' to zero position, 'p' for performance stats")
        print("-" * 70)
        
        last_display_time = time.perf_counter()
        display_interval = 0.05  # 20Hz display update
        
        # Import for non-blocking input (optional)
        import select
        import sys
        import termios
        import tty
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        while True:
            current_time = time.perf_counter()
            
            # Check for keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'z':
                    encoder.zero_position()
                    print("\n*** Position zeroed ***")
                elif key == 'p':
                    stats = encoder.get_performance_stats()
                    print(f"\n*** Performance Stats ***")
                    print(f"Poll rate: {stats['actual_poll_rate']:.0f}Hz (Target: {stats['target_poll_rate']}Hz)")
                    print(f"Efficiency: {stats['poll_efficiency']:.1f}%")
                    print(f"Total polls: {stats['total_polls']:,}")
                    print(f"Errors: {stats['error_count']} ({stats['error_rate']:.3f}%)")
                    print(f"Missed polls: {stats['missed_polls']}")
                    print("-" * 70)
            
            # Update display at fixed rate
            if current_time - last_display_time >= display_interval:
                degrees = encoder.get_position_degrees()
                direction = encoder.get_direction()
                velocity = encoder.get_velocity()
                rpm = encoder.get_rpm()
                acceleration = encoder.get_acceleration()
                
                dir_str = "CW " if direction > 0 else "CCW" if direction < 0 else "---"
                
                # Clear line and print comprehensive status
                print(f"\r{degrees:8.2f}° | {dir_str} | "
                      f"{rpm:6.1f} RPM | {velocity:7.0f} c/s | "
                      f"Accel: {acceleration:7.0f} c/s² ", end='', flush=True)
                
                last_display_time = current_time
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
            
    except KeyboardInterrupt:
        print("\n\nStopping encoder monitoring...")
        
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        # Show final statistics
        stats = encoder.get_performance_stats()
        print(f"\nFinal Statistics:")
        print(f"  Actual poll rate: {stats['actual_poll_rate']:.0f}Hz")
        print(f"  Poll efficiency: {stats['poll_efficiency']:.1f}%")
        print(f"  Total polls: {stats['total_polls']:,}")
        print(f"  Error count: {stats['error_count']}")
        print(f"  Final position: {encoder.get_position_degrees():.2f}°")
        
    finally:
        encoder.cleanup()
        print("Cleanup completed")

if __name__ == "__main__":
    main()