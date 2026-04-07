"""
Superlinked NLQ uses `except instructor.InstructorRetryException`.
Instructor >= 1.11 defines it on `instructor.core`, not the top-level package.
"""

from __future__ import annotations

import instructor
from instructor.core.exceptions import InstructorRetryException

setattr(instructor, "InstructorRetryException", InstructorRetryException)
