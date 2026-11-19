{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% set listed = (['__call__'] if '__call__' in members else []) + methods | reject('equalto', '__init__') | list %}
{% if listed %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
      :nosignatures:
{% for item in listed %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
