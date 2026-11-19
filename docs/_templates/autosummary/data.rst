{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autodata:: {{ objname }}
{% if objname.endswith("_params") %}
   Default ``params`` of :func:`{{ objname[:-7] }}_weights`.  The meaning of
   each key is given in :ref:`default_kernel_params` on the API reference page.
{% endif %}
