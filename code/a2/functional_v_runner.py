"""Register the round5 V backend without changing historical runners."""
import factory
from functional_v_bank import FunctionalVBank
_original = factory.bank_class
factory.bank_class = lambda method: FunctionalVBank if method == 'direction_functional_v' else _original(method)
from run_representation_variants import main

if __name__ == '__main__':
    main()
