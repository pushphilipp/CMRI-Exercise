#!/bin/bash
# Simple activation alias
# Add this to your ~/.bashrc or ~/.bash_profile for convenience:
# alias cmri='cd /path/to/CMRI-Exercise && source activate_env.sh'

# For now, you can create a temporary alias:
echo "Creating temporary alias 'cmri' for easy activation..."
echo "Usage: cmri"
echo ""

# Create alias for current session
alias cmri="cd /home/z0047t6y/git/CMRI-Exercise && source activate_env.sh"

echo "✅ Alias created! Now you can type 'cmri' to activate the environment."
echo ""
echo "To make this permanent, add this line to your ~/.bashrc:"
echo "alias cmri='cd /home/z0047t6y/git/CMRI-Exercise && source activate_env.sh'"