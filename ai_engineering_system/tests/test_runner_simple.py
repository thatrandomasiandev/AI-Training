"""
Simple test runner for the AI Engineering System.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
import json

from ..core.orchestrator import AIEngineeringOrchestrator
from .unit_tests import UnitTestSuite
from .integration_tests import IntegrationTestSuite
from .performance_tests import PerformanceTestSuite
from .validation_tests import ValidationTestSuite


class TestRunner:
    """Simple test runner."""
    
    def __init__(self, ai_system: AIEngineeringOrchestrator):
        self.logger = logging.getLogger(__name__)
        self.ai_system = ai_system
        
        self.test_suites = {
            "unit": UnitTestSuite(ai_system),
            "integration": IntegrationTestSuite(ai_system),
            "performance": PerformanceTestSuite(ai_system),
            "validation": ValidationTestSuite(ai_system)
        }
        
        self.test_results = {}
        self.logger.info("Test runner initialized")
    
    def get_test_suites(self) -> List[str]:
        return list(self.test_suites.keys())
    
    async def run_test_suite(self, suite: str) -> Dict[str, Any]:
        if suite not in self.test_suites:
            return {"success": False, "error": f"Unknown test suite: {suite}"}
        
        self.logger.info(f"Running test suite: {suite}")
        
        test_cases = self.test_suites[suite].get_test_cases()
        results = {}
        start_time = time.time()
        
        for test_case in test_cases:
            result = await self.test_suites[suite].run_test(test_case)
            results[test_case] = result
        
        duration = time.time() - start_time
        total_tests = len(test_cases)
        passed_tests = sum(1 for r in results.values() if r.get("success", False))
        
        summary = {
            "suite": suite,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "duration": duration,
            "results": results
        }
        
        self.test_results[suite] = summary
        return summary
    
    async def run_all_tests(self) -> Dict[str, Any]:
        all_results = {}
        start_time = time.time()
        
        for suite in self.test_suites:
            result = await self.run_test_suite(suite)
            all_results[suite] = result
        
        total_duration = time.time() - start_time
        total_tests = sum(r["total_tests"] for r in all_results.values())
        total_passed = sum(r["passed_tests"] for r in all_results.values())
        
        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "overall_success_rate": total_passed / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration,
            "suite_results": all_results
        }
