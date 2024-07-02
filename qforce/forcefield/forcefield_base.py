class ForcefieldSettings:

    _always_on = ['bond', 'angle']

    __default_settings = {
            'urey': True,
            'cross_bond_bond': False, 
            'cross_bond_angle': False, 
            'cross_angle_angle': False, 
            '_cross_dihed_angle': False, 
            '_cross_dihed_bond': False, 
            'dihedral/rigid': True, 
            'dihedral/improper': True, 
            'dihedral/flexible': True, 
            'dihedral/inversion': True, 
            'dihedral/pitorsion': True, 
            'non_bonded': True, 
            'charge_flux/bond': False, 
            'charge_flux/bond_prime': False, 
            'charge_flux/angle': False, 
            'charge_flux/angle_prime': False, 
            'charge_flux/bond_bond': False, 
            'charge_flux/bond_angle': False, 
            'charge_flux/angle_angle': False, 
            'local_frame/bisector': False, 
            'local_frame/z_then_x': False, 
            'local_frame/z_only': True, 
            'local_frame/z_then_bisector': True, 
            'local_frame/trisector': True,
    }
    
    _settings = {}
    _remove = []

    @classmethod
    def get_questions(cls):
        tpl = '# Turn {key} FF term on or off\n{key} = {default} :: bool\n\n'
        tplset = '# The {key} FF term is alwayson\n{key} = {default} :: bool :: [True]\n\n'
        settings = cls.__default_settings.copy()
        settings.update(cls._settings)
        #
        for key in cls._remove:
            if key in settings:
                del settings[key]
        #
        questions = ''
        for name in cls._always_on:
            questions += tplset.format(key=name, default='True')
        for name, default in settings.items():
            questions += tpl.format(key=name, default=default)
        return questions
