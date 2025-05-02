"""
HR Reports Module - Generates enhanced statistical reports from database data
"""
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class HRReportGenerator:
    def __init__(self, db_instance):
        """Initialize with a database connection"""
        self.db = db_instance
        logger.info("HR Report Generator initialized")
    
    def generate_detailed_hr_report(self, period: str = "last 7 days") -> Dict[str, Any]:
        """
        Generates a comprehensive HR report with statistics and visual analytics
        
        Args:
            period: Time period for the report (e.g., "last 7 days", "last month")
            
        Returns:
            Dictionary containing report data and embedded visualizations
        """
        logger.info(f"Generating detailed HR report for period: {period}")
        
        # Extract basic stats (simulating what would be pulled from the fake_db)
        # In production, this would query a real database
        basic_stats = {
            "period": period,
            "new_hires": 5,
            "leave_requests_processed": 15,
            "pending_decisions": len([d for d in self.db["decisions"].values() if d["status"] == "Pending Validation"]),
            "predicted_attrition_risk_high": 3,
            "predicted_attrition_risk_medium": 7,
            "predicted_attrition_risk_low": 42,
            "report_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add trend analysis
        basic_stats["trends"] = self._calculate_trends()
        
        # Generate and attach visualizations
        basic_stats["visualizations"] = self._generate_visualizations()
        
        # Add department breakdown
        basic_stats["department_stats"] = self._get_department_stats()
        
        # Add action recommendations based on data
        basic_stats["recommendations"] = self._generate_recommendations(basic_stats)
        
        # Log the report generation
        self.db["agent_log"].append({
            "action": "generate_detailed_hr_report", 
            "params": {"period": period}, 
            "timestamp": time.time()
        })
        
        return basic_stats
    
    def get_pending_decisions_report(self) -> Dict[str, Any]:
        """
        Generates a detailed report of all pending decisions with analysis
        
        Returns:
            Dictionary with pending decisions and analysis
        """
        logger.info("Generating pending decisions report")
        
        # Get all pending decisions
        pending_decisions = [d for d in self.db["decisions"].values() if d["status"] == "Pending Validation"]
        
        # Categorize decisions by type
        decision_types = {}
        for decision in pending_decisions:
            action_type = decision["action_name"]
            if action_type not in decision_types:
                decision_types[action_type] = 0
            decision_types[action_type] += 1
        
        # Sort decisions by confidence score
        high_priority = [d for d in pending_decisions if d["confidence_score"] < 0.6]
        medium_priority = [d for d in pending_decisions if 0.6 <= d["confidence_score"] < 0.8]
        low_priority = [d for d in pending_decisions if d["confidence_score"] >= 0.8]
        
        # Calculate average age of pending decisions
        current_time = time.time()
        if pending_decisions:
            avg_age = sum(current_time - d["timestamp"] for d in pending_decisions) / len(pending_decisions)
            avg_age_hours = round(avg_age / 3600, 1)
        else:
            avg_age_hours = 0
        
        report = {
            "total_pending": len(pending_decisions),
            "decision_types": decision_types,
            "priority_breakdown": {
                "high_priority": len(high_priority),
                "medium_priority": len(medium_priority),
                "low_priority": len(low_priority)
            },
            "avg_age_hours": avg_age_hours,
            "oldest_decision_hours": round((current_time - min([d["timestamp"] for d in pending_decisions], default=current_time)) / 3600, 1) if pending_decisions else 0,
            "decisions": pending_decisions,  # Full decision data
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Log the report generation
        self.db["agent_log"].append({
            "action": "get_pending_decisions_report", 
            "params": {}, 
            "timestamp": time.time()
        })
        
        return report
    
    def get_attrition_risk_report(self, department: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a detailed report on attrition risk with analytics
        
        Args:
            department: Optional filter for a specific department
            
        Returns:
            Dictionary with attrition analysis and recommendations
        """
        logger.info(f"Generating attrition risk report for department: {department if department else 'all'}")
        
        # In a real system, this would query actual employee data
        # Here we're simulating the analysis with synthetic data
        
        # Simulated department data
        departments = ["Engineering", "Marketing", "Sales", "Finance", "HR"]
        risk_levels = ["High", "Medium", "Low"]
        
        # Generate synthetic risk distribution
        import random
        random.seed(42)  # For reproducibility
        
        risk_data = {}
        total_high = 0
        total_medium = 0
        total_low = 0
        
        for dept in departments:
            if department and dept != department:
                continue
                
            # Generate random but plausible numbers
            high = random.randint(1, 5)
            medium = random.randint(3, 12)
            low = random.randint(10, 50)
            
            risk_data[dept] = {
                "High": high,
                "Medium": medium,
                "Low": low,
                "Total": high + medium + low
            }
            
            total_high += high
            total_medium += medium
            total_low += low
        
        # Calculate risk factors (simulated)
        risk_factors = [
            {"factor": "Work-life balance concerns", "impact_score": 0.8},
            {"factor": "Limited career advancement", "impact_score": 0.75},
            {"factor": "Compensation below market", "impact_score": 0.7},
            {"factor": "Manager relationship issues", "impact_score": 0.65},
            {"factor": "Skill-role mismatch", "impact_score": 0.6}
        ]
        
        # Generate recommendations
        recommendations = self._generate_attrition_recommendations(risk_data, risk_factors)
        
        # Create report
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "department_filter": department,
            "total_employees_at_risk": {
                "high": total_high,
                "medium": total_medium,
                "low": total_low,
                "total": total_high + total_medium + total_low
            },
            "departmental_breakdown": risk_data,
            "key_risk_factors": risk_factors,
            "recommendations": recommendations,
            "trend": "Increasing" if total_high > 2 else "Stable",
            "historical_comparison": "5% increase in high-risk employees compared to previous quarter"
        }
        
        # Log the report generation
        self.db["agent_log"].append({
            "action": "get_attrition_risk_report", 
            "params": {"department": department}, 
            "timestamp": time.time()
        })
        
        return report
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate trends from historical data (simulated)"""
        return {
            "leave_requests": "+5% from previous period",
            "time_to_approve": "-10% from previous period",
            "average_confidence_score": "0.72 (unchanged)",
            "attrition_risk": "+2% from previous period"
        }
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """Generate base64-encoded visualizations for the report"""
        # This would embed actual charts in the report
        # Here we're just providing placeholders
        
        visualizations = {}
        
        # In a real implementation, this would create actual matplotlib charts
        # and convert them to base64 strings
        
        # Example (commented out to avoid dependencies):
        """
        # Create a simple bar chart
        plt.figure(figsize=(10, 6))
        departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
        values = [23, 17, 35, 12, 18]
        sns.barplot(x=departments, y=values)
        plt.title('Leave Requests by Department')
        plt.tight_layout()
        
        # Convert the plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Encode to base64
        visualizations['leave_by_dept'] = base64.b64encode(image_png).decode('utf-8')
        """
        
        # For demo, just return placeholder strings
        visualizations['leave_by_dept'] = "BASE64_ENCODED_CHART_WOULD_BE_HERE"
        visualizations['attrition_risk_trend'] = "BASE64_ENCODED_CHART_WOULD_BE_HERE"
        
        return visualizations
    
    def _get_department_stats(self) -> Dict[str, Any]:
        """Generate department-level statistics (simulated)"""
        return {
            "Engineering": {
                "headcount": 45,
                "leave_requests": 8,
                "attrition_risk": "Medium",
                "pending_decisions": 3
            },
            "Marketing": {
                "headcount": 22,
                "leave_requests": 4,
                "attrition_risk": "Low",
                "pending_decisions": 1
            },
            "Sales": {
                "headcount": 38,
                "leave_requests": 7,
                "attrition_risk": "High",
                "pending_decisions": 5
            },
            "Finance": {
                "headcount": 15,
                "leave_requests": 2,
                "attrition_risk": "Low",
                "pending_decisions": 0
            },
            "HR": {
                "headcount": 8,
                "leave_requests": 1,
                "attrition_risk": "Low",
                "pending_decisions": 0
            }
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate action recommendations based on the stats"""
        recommendations = []
        
        # Add recommendations based on data
        if stats["pending_decisions"] > 10:
            recommendations.append({
                "priority": "High",
                "action": "Review pending decisions",
                "rationale": f"There are {stats['pending_decisions']} decisions awaiting review, which is above threshold."
            })
        
        if stats["predicted_attrition_risk_high"] > 2:
            recommendations.append({
                "priority": "High",
                "action": "Schedule manager 1:1s with high-risk employees",
                "rationale": f"There are {stats['predicted_attrition_risk_high']} employees at high risk of attrition."
            })
        
        if stats.get("trends", {}).get("leave_requests", "").startswith("+"):
            recommendations.append({
                "priority": "Medium",
                "action": "Review team capacity planning",
                "rationale": "Leave requests are trending upward. Ensure adequate coverage."
            })
        
        # Always add some generic recommendations if list is empty
        if not recommendations:
            recommendations.append({
                "priority": "Medium",
                "action": "Review department workload distribution",
                "rationale": "Regular workload reviews help identify burnout risks early."
            })
        
        return recommendations
    
    def _generate_attrition_recommendations(self, risk_data: Dict[str, Dict[str, int]], 
                                         risk_factors: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate recommendations for attrition risk mitigation"""
        recommendations = []
        
        # Find department with highest risk
        high_risk_depts = []
        for dept, data in risk_data.items():
            if data["High"] >= 3:
                high_risk_depts.append(dept)
        
        if high_risk_depts:
            recommendations.append({
                "priority": "High",
                "action": f"Conduct stay interviews in {', '.join(high_risk_depts)}",
                "rationale": f"These departments have 3+ employees at high attrition risk."
            })
        
        # Add recommendations based on top risk factors
        top_factor = risk_factors[0]["factor"]
        recommendations.append({
            "priority": "High",
            "action": f"Address '{top_factor}' through targeted initiatives",
            "rationale": f"This is the highest-impact attrition factor (impact score: {risk_factors[0]['impact_score']})."
        })
        
        # Add general recommendation
        recommendations.append({
            "priority": "Medium",
            "action": "Review compensation benchmarks",
            "rationale": "Regular market comparisons help prevent compensation-related attrition."
        })
        
        return recommendations