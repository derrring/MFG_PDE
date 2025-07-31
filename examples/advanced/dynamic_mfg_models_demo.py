"""
Dynamic MFG Model Instantiation Demo.

This example demonstrates how to use the MFG model factory system to 
dynamically create and experiment with different MFG problem types
from configuration.
"""

import numpy as np
import sys
from pathlib import Path

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mfg_pde.factory import (
    MFGModelFactory, 
    get_mfg_factory,
    create_mfg_model,
    list_mfg_models,
    MFGModelSpec
)
from mfg_pde.config import OmegaConfManager
from mfg_pde.utils.logging import get_logger, configure_research_logging
from mfg_pde.visualization import MFGAnalytics

# Configure logging
configure_research_logging("dynamic_mfg_demo", level="INFO")
logger = get_logger(__name__)


def demo_available_models():
    """Demonstrate listing and exploring available MFG models."""
    logger.info(" Demo 1: Available MFG Models")
    
    # Get factory instance
    factory = get_mfg_factory()
    
    # List all available models
    models = list_mfg_models()
    logger.info(f"Available models: {models}")
    
    # Get detailed information about each model
    for model_name in models:
        try:
            info = factory.get_model_info(model_name)
            logger.info(f"\n {model_name.upper()}:")
            logger.info(f"  Description: {info['description']}")
            logger.info(f"  Category: {info['category']}")
            logger.info(f"  Physics: {info['physics_type']}")
            logger.info(f"  Templates: {info['available_templates']}")
        except Exception as e:
            logger.warning(f"Could not get info for {model_name}: {e}")
    
    return models


def demo_model_creation_from_templates():
    """Demonstrate creating models from built-in templates."""
    logger.info("🏭 Demo 2: Model Creation from Templates")
    
    factory = get_mfg_factory()
    
    # Create different models from templates
    models_to_test = [
        ("example", "basic"),
        ("crowd_dynamics", "corridor"),
        ("financial_market", "basic_trading"),
        ("traffic_flow", "highway"),
        ("epidemic_spread", "basic_epidemic")
    ]
    
    created_models = {}
    
    for model_type, template in models_to_test:
        try:
            logger.info(f"Creating {model_type} from template '{template}'...")
            
            # Create model from template
            model = factory.create_from_template(model_type, template)
            created_models[f"{model_type}_{template}"] = model
            
            # Log basic info
            logger.info(f"  SUCCESS: Created: {type(model).__name__}")
            logger.info(f"  Domain: [{model.xmin:.1f}, {model.xmax:.1f}] with {model.Nx} points")
            logger.info(f"  Time: [0, {model.T:.1f}] with {model.Nt} steps")
            
            # Model-specific info
            if hasattr(model, 'congestion_power'):
                logger.info(f"  Congestion power: {model.congestion_power}")
            if hasattr(model, 'price_impact'):
                logger.info(f"  Price impact: {model.price_impact}")
            if hasattr(model, 'infection_rate'):
                logger.info(f"  Infection rate: {model.infection_rate}")
                
        except Exception as e:
            logger.error(f"  ERROR: Failed to create {model_type}: {e}")
    
    return created_models


def demo_custom_model_configuration():
    """Demonstrate creating models with custom configurations."""
    logger.info("⚙️ Demo 3: Custom Model Configuration")
    
    factory = get_mfg_factory()
    
    # Custom crowd dynamics configuration
    crowd_config = {
        "xmin": 0.0, "xmax": 20.0, "Nx": 201,
        "T": 10.0, "Nt": 201,
        "sigma": 0.05,
        "coefCT": 1.5,
        "congestion_power": 3.0,
        "target_velocity": 1.2,
        "panic_factor": 0.2,
        "destination_attraction": 3.0
    }
    
    # Custom financial market configuration
    finance_config = {
        "xmin": -5.0, "xmax": 5.0, "Nx": 101,
        "T": 0.5, "Nt": 101,
        "sigma": 0.1,
        "coefCT": 0.8,
        "price_impact": 1.5,
        "risk_aversion": 2.0,
        "volatility": 0.4,
        "transaction_cost": 0.005
    }
    
    # Create custom models
    custom_models = {}
    
    try:
        logger.info("Creating custom crowd dynamics model...")
        crowd_model = factory.create_model("crowd_dynamics", crowd_config)
        custom_models["custom_crowd"] = crowd_model
        logger.info(f"  SUCCESS: Custom crowd model: panic_factor={crowd_model.panic_factor}")
        
        logger.info("Creating custom financial market model...")
        finance_model = factory.create_model("financial_market", finance_config) 
        custom_models["custom_finance"] = finance_model
        logger.info(f"  SUCCESS: Custom finance model: volatility={finance_model.volatility}")
        
    except Exception as e:
        logger.error(f"ERROR: Custom model creation failed: {e}")
    
    return custom_models


def demo_configuration_file_loading():
    """Demonstrate loading models from configuration files."""
    logger.info("📄 Demo 4: Configuration File Loading")
    
    # Create sample configuration
    config_data = {
        "model_type": "epidemic_spread",
        "model": {
            "xmin": 0.0, "xmax": 15.0, "Nx": 151,
            "T": 150.0, "Nt": 301,
            "sigma": 0.3,
            "infection_rate": 0.4,
            "vaccination_rate": 0.08,
            "health_cost": 20.0,
            "social_distancing_cost": 2.0
        },
        "solver": {
            "type": "fast_solver",
            "max_iterations": 50,
            "tolerance": 1e-5
        }
    }
    
    # Save configuration to file
    config_file = Path("./temp_mfg_config.yaml")
    config_manager = OmegaConfManager(config_data)
    config_manager.save_config(str(config_file))
    
    try:
        # Load model from configuration file
        factory = get_mfg_factory()
        model = factory.create_from_config_file(str(config_file))
        
        logger.info("SUCCESS: Model loaded from configuration file")
        logger.info(f"  Model type: {type(model).__name__}")
        logger.info(f"  Infection rate: {model.infection_rate}")
        logger.info(f"  Vaccination rate: {model.vaccination_rate}")
        
        # Clean up
        config_file.unlink()
        
        return model
        
    except Exception as e:
        logger.error(f"ERROR: Config file loading failed: {e}")
        if config_file.exists():
            config_file.unlink()
        return None


def demo_model_validation():
    """Demonstrate model configuration validation."""
    logger.info("SUCCESS: Demo 5: Model Configuration Validation")
    
    factory = get_mfg_factory()
    
    # Test valid configuration
    valid_config = {
        "xmin": 0.0, "xmax": 1.0, "Nx": 51,
        "T": 1.0, "Nt": 51,
        "sigma": 0.5,
        "price_impact": 0.3
    }
    
    # Test invalid configuration (missing required params)
    invalid_config = {
        "sigma": 0.5,
        "price_impact": 0.3
        # Missing required parameters
    }
    
    # Validate configurations
    logger.info("Validating valid configuration...")
    validation = factory.validate_config("financial_market", valid_config)
    logger.info(f"  Valid: {validation['valid']}")
    if validation['warnings']:
        logger.info(f"  Warnings: {validation['warnings']}")
    
    logger.info("Validating invalid configuration...")
    validation = factory.validate_config("financial_market", invalid_config)
    logger.info(f"  Valid: {validation['valid']}")
    if validation['errors']:
        logger.info(f"  Errors: {validation['errors']}")
    if validation['missing_required']:
        logger.info(f"  Missing required: {validation['missing_required']}")


def demo_custom_model_registration():
    """Demonstrate registering custom MFG model types."""
    logger.info(" Demo 6: Custom Model Registration")
    
    # Create a custom MFG model class
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.utils.aux_func import npart, ppart
    from mfg_pde.core.mfg_problem import VALUE_BEFORE_SQUARE_LIMIT
    
    class CustomEconomyMFG(MFGProblem):
        """Custom economy MFG model."""
        
        def __init__(self, market_power=1.0, regulation_strength=0.5, **kwargs):
            super().__init__(**kwargs)
            self.market_power = market_power
            self.regulation_strength = regulation_strength
        
        def H(self, x_idx, m_at_x, p_values, t_idx=None):
            """Simple Hamiltonian for demonstration."""
            p_forward = p_values.get("forward")
            p_backward = p_values.get("backward")

            if p_forward is None or p_backward is None:
                return np.nan
            if (np.isnan(p_forward) or np.isinf(p_forward) or
                np.isnan(p_backward) or np.isinf(p_backward) or
                np.isnan(m_at_x) or np.isinf(m_at_x)):
                return np.nan

            npart_val_fwd = npart(p_forward)
            ppart_val_bwd = ppart(p_backward)

            if (abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT or 
                abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT):
                return np.nan

            try:
                term_npart_sq = npart_val_fwd**2
                term_ppart_sq = ppart_val_bwd**2
            except OverflowError:
                return np.nan

            if (np.isinf(term_npart_sq) or np.isnan(term_npart_sq) or
                np.isinf(term_ppart_sq) or np.isnan(term_ppart_sq)):
                return np.nan

            hamiltonian_control_part = 0.5 * self.coefCT * (term_npart_sq + term_ppart_sq)
            potential_cost = self.f_potential[x_idx]
            
            try:
                market_coupling = self.market_power * m_at_x**2
                regulation_cost = self.regulation_strength * m_at_x
            except OverflowError:
                return np.nan

            result = hamiltonian_control_part - potential_cost - market_coupling - regulation_cost
            
            if np.isinf(result) or np.isnan(result):
                return np.nan
            return result
        
        def dH_dm(self, x_idx, m_at_x, p_values, t_idx=None):
            """Derivative for custom model."""
            if np.isnan(m_at_x) or np.isinf(m_at_x):
                return np.nan
            try:
                return 2.0 * self.market_power * m_at_x + self.regulation_strength
            except OverflowError:
                return np.nan
    
    # Create custom model specification
    custom_spec = MFGModelSpec(
        name="custom_economy",
        description="Custom economy MFG model with market power and regulation",
        model_class=CustomEconomyMFG,
        category="custom",
        dimension=1,
        physics_type="economy",
        required_params=["xmin", "xmax", "Nx", "T", "Nt"],
        optional_params={
            "sigma": 1.0,
            "coefCT": 0.5,
            "market_power": 1.0,
            "regulation_strength": 0.5
        },
        geometry_support=["1d_uniform"],
        examples={
            "free_market": {
                "xmin": 0.0, "xmax": 1.0, "Nx": 51,
                "T": 1.0, "Nt": 51,
                "market_power": 2.0, "regulation_strength": 0.1
            },
            "regulated": {
                "xmin": 0.0, "xmax": 1.0, "Nx": 51,
                "T": 1.0, "Nt": 51,
                "market_power": 0.5, "regulation_strength": 1.0
            }
        }
    )
    
    # Register custom model
    from mfg_pde.factory import register_custom_model
    register_custom_model(custom_spec)
    
    logger.info("SUCCESS: Registered custom economy MFG model")
    
    # Test creating custom model
    try:
        custom_model = create_mfg_model("custom_economy", 
                                       xmin=0, xmax=1, Nx=51, T=1, Nt=51,
                                       market_power=1.5, regulation_strength=0.3)
        logger.info(f"SUCCESS: Created custom model: market_power={custom_model.market_power}")
        return custom_model
    except Exception as e:
        logger.error(f"ERROR: Custom model creation failed: {e}")
        return None


def demo_model_search_and_filtering():
    """Demonstrate searching and filtering models."""
    logger.info(" Demo 7: Model Search and Filtering")
    
    factory = get_mfg_factory()
    registry = factory.registry
    
    # Search by category
    finance_models = registry.list_models("finance")
    logger.info(f"Finance models: {finance_models}")
    
    # Search by dimension
    models_1d = registry.get_models_by_dimension(1)
    logger.info(f"1D models: {models_1d}")
    
    # Search by geometry support
    complex_geom_models = registry.get_models_by_geometry("2d_complex")
    logger.info(f"Complex geometry models: {complex_geom_models}")
    
    # Advanced search
    congestion_models = registry.search_models(physics_type="congestion")
    logger.info(f"Congestion models: {congestion_models}")
    
    application_models = registry.search_models(category="applications")
    logger.info(f"Application models: {application_models}")


def main():
    """Run all dynamic MFG model demonstrations."""
    logger.info(" Starting Dynamic MFG Model Instantiation Demo")
    
    try:
        # Run all demonstrations
        models = demo_available_models()
        created_models = demo_model_creation_from_templates()
        custom_models = demo_custom_model_configuration()
        config_model = demo_configuration_file_loading()
        demo_model_validation()
        custom_registered = demo_custom_model_registration()
        demo_model_search_and_filtering()
        
        # Summary
        total_models = len(created_models) + len(custom_models) + (1 if config_model else 0) + (1 if custom_registered else 0)
        logger.info(f"\n Demo completed successfully!")
        logger.info(f" Created {total_models} different MFG model instances")
        logger.info(f" Available model types: {len(models)}")
        
        # Example of using a created model (if any exist)
        if created_models:
            example_model = list(created_models.values())[0]
            logger.info(f"\n🔬 Example model ready for solving:")
            logger.info(f"   Type: {type(example_model).__name__}")
            logger.info(f"   Domain: [{example_model.xmin}, {example_model.xmax}]")
            logger.info(f"   Resolution: {example_model.Nx} × {example_model.Nt}")
        
    except Exception as e:
        logger.error(f"ERROR: Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()