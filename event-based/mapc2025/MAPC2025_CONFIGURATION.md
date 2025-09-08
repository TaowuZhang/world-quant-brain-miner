# MAPC2025 Competition Configuration

## Overview
The enhanced template generator has been optimized for the MAPC2025 competition with the following key changes:

## Competition Requirements
- **Region**: GLB (Global) only
- **Neutralization**: REVERSION_AND_MOMENTUM (Risk-Adjusted Market) - primary focus
- **Delay**: 1 (fixed)
- **Universe**: TOP3000

## Key Changes Made

### 1. Region Configuration
- **Before**: Multiple regions (USA, GLB, EUR, ASI, CHN)
- **After**: GLB region only
- **Impact**: All template generation and simulation focuses on global market

### 2. Neutralization Settings
- **Before**: INDUSTRY as default
- **After**: REVERSION_AND_MOMENTUM as default and primary option
- **Available options for GLB**: REVERSION_AND_MOMENTUM, INDUSTRY, SUBINDUSTRY, SECTOR, COUNTRY, NONE
- **Impact**: Templates optimized for risk-adjusted market strategies

### 3. Delay Configuration
- **Before**: Variable delays (0 and 1) with weighted selection
- **After**: Fixed delay=1 for all GLB simulations
- **Impact**: Consistent delay=1 as required by MAPC2025

### 4. Pyramid Multipliers
- **Before**: Multiple regions with different delay multipliers
- **After**: GLB delay=1 with high priority (2.0 multiplier)
- **Impact**: Maximum focus on the competition requirements

### 5. Simulation Settings
- **Default region**: GLB
- **Default delay**: 1
- **Default neutralization**: REVERSION_AND_MOMENTUM
- **Impact**: All simulations automatically use competition parameters

### 6. Template Generation Prompts
- Updated to mention MAPC2025 competition focus
- Emphasizes REVERSION_AND_MOMENTUM neutralization for risk-adjusted strategies
- Highlights GLB region and delay=1 requirements

## Usage
The generator will now automatically:
1. Focus only on GLB region
2. Use delay=1 for all simulations
3. Prioritize REVERSION_AND_MOMENTUM neutralization
4. Generate templates optimized for risk-adjusted market strategies
5. Test variations with different neutralization settings (REVERSION_AND_MOMENTUM, INDUSTRY, etc.)

## Competition Advantage
- **Focused approach**: No wasted resources on non-competition regions
- **REVERSION_AND_MOMENTUM optimization**: Templates designed for risk-adjusted market strategies
- **Consistent parameters**: All simulations use competition-required settings
- **Efficient resource usage**: Maximum concurrent simulations on relevant parameters only

## Files Modified
- `enhanced_template_generator_v2.py`: Main configuration changes
- All changes are backward compatible and focused on MAPC2025 requirements
