"""
Circuit breaker pattern implementation for FastText operations.

This module implements a circuit breaker pattern to handle repeated FastText failures
gracefully. When FastText operations fail repeatedly, the circuit breaker opens and
prevents further attempts for a cooldown period, falling back to Unicode detection.
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from .logger import get_fasttext_logger

# Use dedicated FastText logger
logger = get_fasttext_logger("circuit_breaker")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, FastText enabled
    OPEN = "open"          # FastText disabled due to failures
    HALF_OPEN = "half_open"  # Testing recovery with limited attempts


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    # Failure threshold to open circuit
    failure_threshold: int = 5
    
    # Time window for counting failures (seconds)
    failure_window: int = 300  # 5 minutes
    
    # Cooldown period before attempting recovery (seconds)
    recovery_timeout: int = 60  # 1 minute
    
    # Success threshold to close circuit from half-open state
    success_threshold: int = 3
    
    # Maximum number of test attempts in half-open state
    half_open_max_attempts: int = 5


class CircuitBreaker:
    """
    Circuit breaker for FastText operations with automatic recovery.
    
    The circuit breaker follows this state machine:
    
    CLOSED -> OPEN: When failure_threshold failures occur within failure_window
    OPEN -> HALF_OPEN: After recovery_timeout seconds
    HALF_OPEN -> CLOSED: After success_threshold consecutive successes
    HALF_OPEN -> OPEN: If any failure occurs or max attempts exceeded
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker with configuration.
        
        Args:
            config: Circuit breaker configuration. Uses defaults if None.
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        
        # Failure tracking
        self._failure_count = 0
        self._failure_timestamps = []
        self._last_failure_time = 0.0
        
        # Recovery tracking
        self._circuit_opened_time = 0.0
        self._half_open_attempts = 0
        self._half_open_successes = 0
        
        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_fallbacks = 0
        self._state_transitions = {
            CircuitState.CLOSED: 0,
            CircuitState.OPEN: 0,
            CircuitState.HALF_OPEN: 0
        }
        
        logger.info(f"Circuit breaker initialized: {self.config}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result if successful
            
        Raises:
            CircuitBreakerOpenError: If circuit is open and fallback should be used
            Exception: Any exception raised by the function
        """
        self._total_calls += 1
        
        # Check circuit state and decide whether to proceed
        if not self._should_attempt_call():
            self._total_fallbacks += 1
            raise CircuitBreakerOpenError("Circuit breaker is open, use fallback")
        
        try:
            # Attempt the function call
            result = func(*args, **kwargs)
            
            # Record success
            self._record_success()
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(e)
            raise
    
    def _should_attempt_call(self) -> bool:
        """
        Determine if a call should be attempted based on circuit state.
        
        Returns:
            True if call should be attempted, False if circuit is open
        """
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self._circuit_opened_time >= self.config.recovery_timeout:
                self._transition_to_half_open()
                return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited attempts in half-open state
            if self._half_open_attempts < self.config.half_open_max_attempts:
                return True
            else:
                # Too many attempts, go back to open
                self._transition_to_open("Max half-open attempts exceeded")
                return False
        
        return False
    
    def _record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            
            # Check if we have enough successes to close the circuit
            if self._half_open_successes >= self.config.success_threshold:
                self._transition_to_closed()
        
        # Clean up old failure timestamps
        self._cleanup_old_failures()
    
    def _record_failure(self, exception: Exception) -> None:
        """
        Record a failed operation.
        
        Args:
            exception: The exception that caused the failure
        """
        current_time = time.time()
        self._total_failures += 1
        self._last_failure_time = current_time
        
        # Add failure timestamp
        self._failure_timestamps.append(current_time)
        
        # Clean up old failures outside the window
        self._cleanup_old_failures()
        
        # Update failure count
        self._failure_count = len(self._failure_timestamps)
        
        logger.warning(f"Circuit breaker recorded failure: {exception}")
        
        if self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open(f"Failure threshold reached: {self._failure_count}")
        
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self._transition_to_open("Failure during half-open state")
    
    def _cleanup_old_failures(self) -> None:
        """Remove failure timestamps outside the failure window."""
        current_time = time.time()
        cutoff_time = current_time - self.config.failure_window
        
        # Remove old timestamps
        self._failure_timestamps = [
            timestamp for timestamp in self._failure_timestamps
            if timestamp > cutoff_time
        ]
    
    def _transition_to_open(self, reason: str) -> None:
        """
        Transition circuit to OPEN state.
        
        Args:
            reason: Reason for opening the circuit
        """
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker OPENING: {reason}")
            self.state = CircuitState.OPEN
            self._circuit_opened_time = time.time()
            self._state_transitions[CircuitState.OPEN] += 1
            
            # Reset half-open counters
            self._half_open_attempts = 0
            self._half_open_successes = 0
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state for recovery testing."""
        if self.state != CircuitState.HALF_OPEN:
            logger.info("Circuit breaker transitioning to HALF_OPEN for recovery testing")
            self.state = CircuitState.HALF_OPEN
            self._state_transitions[CircuitState.HALF_OPEN] += 1
            
            # Reset half-open counters
            self._half_open_attempts = 0
            self._half_open_successes = 0
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state (normal operation)."""
        if self.state != CircuitState.CLOSED:
            logger.info("Circuit breaker CLOSING - recovery successful")
            self.state = CircuitState.CLOSED
            self._state_transitions[CircuitState.CLOSED] += 1
            
            # Reset failure tracking
            self._failure_count = 0
            self._failure_timestamps.clear()
            
            # Reset half-open counters
            self._half_open_attempts = 0
            self._half_open_successes = 0
    
    def is_open(self) -> bool:
        """
        Check if circuit is open (FastText should not be attempted).
        
        Returns:
            True if circuit is open, False otherwise
        """
        return self.state == CircuitState.OPEN
    
    def is_closed(self) -> bool:
        """
        Check if circuit is closed (normal operation).
        
        Returns:
            True if circuit is closed, False otherwise
        """
        return self.state == CircuitState.CLOSED
    
    def is_half_open(self) -> bool:
        """
        Check if circuit is half-open (recovery testing).
        
        Returns:
            True if circuit is half-open, False otherwise
        """
        return self.state == CircuitState.HALF_OPEN
    
    def force_open(self, reason: str = "Manual override") -> None:
        """
        Manually force circuit to open state.
        
        Args:
            reason: Reason for forcing open
        """
        logger.warning(f"Circuit breaker manually forced OPEN: {reason}")
        self._transition_to_open(reason)
    
    def force_close(self, reason: str = "Manual override") -> None:
        """
        Manually force circuit to closed state.
        
        Args:
            reason: Reason for forcing closed
        """
        logger.info(f"Circuit breaker manually forced CLOSED: {reason}")
        self._transition_to_closed()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive circuit breaker statistics.
        
        Returns:
            Dictionary with circuit breaker metrics and status
        """
        current_time = time.time()
        
        # Calculate failure rate
        failure_rate = (self._total_failures / self._total_calls * 100) if self._total_calls > 0 else 0.0
        
        # Calculate fallback rate
        fallback_rate = (self._total_fallbacks / self._total_calls * 100) if self._total_calls > 0 else 0.0
        
        # Time since last failure
        time_since_last_failure = current_time - self._last_failure_time if self._last_failure_time > 0 else None
        
        # Time since circuit opened
        time_since_opened = current_time - self._circuit_opened_time if self._circuit_opened_time > 0 else None
        
        return {
            "state": self.state.value,
            "is_open": self.is_open(),
            "is_closed": self.is_closed(),
            "is_half_open": self.is_half_open(),
            "configuration": {
                "failure_threshold": self.config.failure_threshold,
                "failure_window": self.config.failure_window,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "half_open_max_attempts": self.config.half_open_max_attempts,
            },
            "counters": {
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_fallbacks": self._total_fallbacks,
                "current_failure_count": self._failure_count,
                "half_open_attempts": self._half_open_attempts,
                "half_open_successes": self._half_open_successes,
            },
            "rates": {
                "failure_rate_percent": round(failure_rate, 2),
                "fallback_rate_percent": round(fallback_rate, 2),
            },
            "timing": {
                "time_since_last_failure_seconds": time_since_last_failure,
                "time_since_opened_seconds": time_since_opened,
                "failure_window_seconds": self.config.failure_window,
                "recovery_timeout_seconds": self.config.recovery_timeout,
            },
            "state_transitions": self._state_transitions.copy(),
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        logger.info("Circuit breaker reset to initial state")
        
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._failure_timestamps.clear()
        self._last_failure_time = 0.0
        self._circuit_opened_time = 0.0
        self._half_open_attempts = 0
        self._half_open_successes = 0
        
        # Reset statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_fallbacks = 0
        self._state_transitions = {state: 0 for state in CircuitState}


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and operation should fall back."""
    pass